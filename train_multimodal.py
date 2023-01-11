import datetime
import argparse
import json
import torch
import logging
from tqdm import tqdm
from ml_collections import config_dict
from timeit import default_timer as timer
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
)
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from script.utils import (
    logging_related,
    parse_arguments,
)
from data.data_process import prep_dataset, prep_dataset_, prep_dataloader
from model.model import LMetalSiteMultiModal, LMetalSiteMultiModalBase

# from libauc.losses import APLoss
from libauc.optimizers import SOAP

# from libauc.metrics import auc_prc_score
from torchvision.ops import sigmoid_focal_loss
from model.loss_ import APLoss

# from model.sampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader
from libauc.sampler import DualSampler


def main(conf):
    RANDOM_SEED = int(conf.general.seed)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    device = (
        torch.device("cuda:{}".format(conf.general.gpu_id))
        if torch.cuda.is_available()
        else "cpu"
    )
    logging.info(
        "Training begins at {}".format(datetime.datetime.now().strftime("%m-%d %H:%M"))
    )
    conf.model.feature_dim = 1
    # Load LMetalSite model
    if conf.model.name == "base":
        model = LMetalSiteMultiModalBase(conf.model).to(device)
    elif conf.model.name == "cross_attention":
        model = LMetalSiteMultiModal(conf.model).to(device)
    else:
        raise NotImplementedError("Invalid model name")
    if conf.training.pretrained_encoder:
        checkpoint = torch.load(conf.training.pretrained_encoder)
        logging.info("load encoder from {}".format(conf.training.pretrained_encoder))
        model.params.encoder.load_state_dict(checkpoint["encoder_state"])

    if conf.training.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            [
                {"params": model.params.classifier.parameters()},
                {
                    "params": model.params.encoder.parameters(),
                    "lr": conf.training.encoder_learning_rate,
                },
            ],
            betas=(0.9, 0.99),
            lr=conf.training.learning_rate,
            weight_decay=conf.training.weight_decay,
        )
    else:
        # optimizer = torch.optim.AdamW(
        #     [
        #         {"params": model.params.classifier.parameters()},
        #         {
        #             "params": model.params.encoder.parameters(),
        #             "lr": conf.training.encoder_learning_rate,
        #         },
        #     ],
        #     lr=conf.training.learning_rate,
        #     weight_decay=conf.training.weight_decay,
        # )
        optimizer = SOAP(
            [
                {"params": model.params.classifier.parameters()},
                {
                    "params": model.params.encoder.parameters(),
                    "lr": conf.training.encoder_learning_rate,
                },
            ],
            mode="adam",
            lr=conf.training.learning_rate,
            weight_decay=conf.training.weight_decay,
        )
    metric_auc = BinaryAUROC(thresholds=None)
    metric_auprc = BinaryAveragePrecision(thresholds=None)
    model.training = True  # adding Gaussian noise to embedding
    ligand_list = ["DNA", "RNA", "MG", "CA", "MN", "ZN"]
    # ligand_list = ["MG", "CA", "MN", "ZN"]
    pos_lens = []
    pos_weights = []
    dataloader_train, dataloader_val = [], []
    for ligand in ligand_list:
        train_dataset, val_dataset, pos_weight = prep_dataset_(
            conf, device, ligand=ligand
        )
        sampler = DualSampler(
            train_dataset, conf.training.batch_size, sampling_rate=0.5
        )
        pos_lens.append(sampler.pos_len)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=conf.training.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=0,
            sampler=sampler,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=conf.training.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=0,
        )
        dataloader_train.append(train_dataloader)
        dataloader_val.append(val_dataloader)
        pos_weights.append(pos_weight)
        # pos_lens.append(pos_len)

    for epoch in range(conf.training.epochs):
        for k, ligand in enumerate(ligand_list):
            # loss_func = torch.nn.BCEWithLogitsLoss(
            #     pos_weight=torch.sqrt(torch.tensor(pos_weights[k]))
            # )
            # loss_func = torch.nn.BCEWithLogitsLoss()
            loss_func = APLoss(
                pos_len=pos_lens[k],
                margin=conf.training.margin,
                gamma=conf.training.gamma,
                device=device,
            )
            # loss_func = sigmoid_focal_loss()
            # loss_func = APLoss().to(device)
            logging.info("Training for {}".format(ligand))
            model.ligand = ligand
            model.train()
            train_loss = 0.0
            all_outputs, all_labels = [], []
            for i, batch_data in tqdm(enumerate(dataloader_train[k])):
                index, feats_1, feats_2, labels = batch_data
                optimizer.zero_grad(set_to_none=True)
                feats_1 = feats_1.to(device)
                feats_2 = feats_2.to(device)
                labels = labels.to(device)
                outputs = model(feats_1, feats_2)
                # loss_ = sigmoid_focal_loss(
                #     outputs * masks, labels, gamma=2, reduction="mean"
                # )
                loss_ = loss_func(outputs, labels, index)
                loss_.backward()
                optimizer.step()
                all_outputs.append(torch.sigmoid(outputs))
                all_labels.append(labels)
                train_loss += loss_.detach().cpu().numpy()

            all_outputs = torch.cat(all_outputs).detach().cpu()
            all_labels = torch.cat(all_labels).detach().cpu()
            train_auc = metric_auc(all_outputs, all_labels)
            train_auprc = metric_auprc(all_outputs, all_labels)
            logging.info(
                "Epoch {} train loss {:.4f}, auc {:.3f}, auprc: {:.3f}".format(
                    epoch + 1,
                    train_loss / (i + 1),
                    train_auc,
                    train_auprc,
                )
            )
            writer.add_scalars(
                "train_{}".format(ligand),
                {
                    "loss": train_loss / (i + 1),
                    "auc": train_auc,
                    "auprc": train_auprc,
                },
                epoch + 1,
            )
            model.eval()
            with torch.no_grad():
                model.training = False
                val_loss = 0.0
                all_outputs, all_labels = [], []
                for i, batch_data in tqdm(enumerate(dataloader_val[k])):
                    _, feats_1, feats_2, labels = batch_data
                    feats_1 = feats_1.to(device)
                    feats_2 = feats_2.to(device)
                    labels = labels.to(device)
                    outputs = model(feats_1, feats_2)
                    # loss_ = sigmoid_focal_loss(
                    #     outputs * masks, labels, gamma=2, reduction="mean"
                    # )
                    # loss_ = loss_func(outputs, labels, index)
                    all_outputs.append(torch.sigmoid(outputs))
                    all_labels.append(labels)
                    # val_loss += loss_.detach().cpu().numpy()

                all_outputs = torch.cat(all_outputs).detach().cpu()
                all_labels = torch.cat(all_labels).detach().cpu()
                val_auc = metric_auc(all_outputs, all_labels)
                val_auprc = metric_auprc(all_outputs, all_labels)
                logging.info(
                    "Epoch {} val auc {:.3f}, auprc: {:.3f}".format(
                        epoch + 1,
                        val_auc,
                        val_auprc,
                    )
                )
                writer.add_scalars(
                    "val_{}".format(ligand),
                    {
                        "auc": val_auc,
                        "auprc": val_auprc,
                    },
                    epoch + 1,
                )

    logging.info(
        "Training is done at {}".format(datetime.datetime.now().strftime("%m-%d %H:%M"))
    )


if __name__ == "__main__":
    start = timer()
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    with open(args.config, "r") as f:
        conf = json.load(f)

    output_path = None
    if not conf["general"]["debug"]:
        output_path = (
            Path("./results/")
            / Path(args.config).stem
            / Path(conf["data"]["feature"] + "_" + conf["model"]["name"])
            / Path(
                str(datetime.datetime.now())[:16].replace(" ", "-").replace(":", "-")
            )
        )
        output_path.mkdir(parents=True, exist_ok=True)
        conf["output_path"] = "./" + str(output_path)
        with open(str(output_path) + "/config.json", "w") as f:
            json.dump(conf, f, indent=4)

    conf = config_dict.ConfigDict(conf)
    """
    logging related part
    """
    logging_related(output_path=output_path, debug=conf.general.debug)
    writer = SummaryWriter(log_dir=output_path)
    main(conf)
    writer.flush()
    end = timer()
    logging.info("Total time used: {:.1f}".format(end - start))
