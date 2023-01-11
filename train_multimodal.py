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
from data.data_process import prep_dataset_
from model.model import LMetalSiteMultiModal, LMetalSiteMultiModalBase
from libauc.optimizers import SOAP, PESG
from torchvision.ops import sigmoid_focal_loss
from model.loss_ import APLoss
from libauc.losses import AUCMLoss
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

    optimizer_1 = torch.optim.AdamW(
        [
            {"params": model.params.classifier.parameters()},
            {
                "params": model.params.encoder.parameters(),
                "lr": conf.training.encoder_learning_rate,
            },
        ],
        lr=conf.training.learning_rate,
        weight_decay=conf.training.weight_decay_adamw,
    )

    # optimizer_2 = SOAP(
    #     [
    #         {"params": model.params.classifier.parameters()},
    #         {
    #             "params": model.params.encoder.parameters(),
    #             "lr": conf.training.encoder_learning_rate,
    #         },
    #     ],
    #     mode="adam",
    #     lr=conf.training.learning_rate,
    #     weight_decay=conf.training.weight_decay_auprc,
    # )
    loss_fn = AUCMLoss(device=device)
    optimizer_2 = PESG(model, 
                 loss_fn=loss_fn,
                 lr=conf.training.learning_rate, 
                 momentum=0.9,
                 margin=conf.training.margin, 
                 epoch_decay=3e-3, 
                 device=device,
                 weight_decay=conf.training.weight_decay_auprc)
     
    metric_auc = BinaryAUROC(thresholds=None)
    metric_auprc = BinaryAveragePrecision(thresholds=None)
    model.training = True  # adding Gaussian noise to embedding
    ligand_list = ["DNA", "RNA", "MG", "CA", "MN", "ZN"]
    pos_weights = []
    train_datasets, val_datasets = [], []
    for ligand in ligand_list:
        train_dataset, val_dataset, pos_weight = prep_dataset_(
            conf, device, ligand=ligand
        )
        pos_weights.append(pos_weight)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    for epoch in range(conf.training.epochs):
        for k, ligand in enumerate(ligand_list):
            if epoch < conf.training.warm_up_epochs:
                loss_func = torch.nn.BCEWithLogitsLoss(
                    pos_weight=torch.sqrt(torch.tensor(pos_weights[k]))
                )
                # loss_func = torch.nn.BCEWithLogitsLoss()
                dataloader_train = DataLoader(
                    train_datasets[k],
                    batch_size=conf.training.batch_size,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True,
                    num_workers=0,
                )
            else:
                sampler = DualSampler(
                    train_datasets[k], conf.training.batch_size, sampling_rate=0.5
                )
                # loss_func = APLoss(
                #     pos_len=sampler.pos_len,
                #     margin=conf.training.margin,
                #     gamma=conf.training.gamma,
                #     device=device,
                # )
                loss_fn = AUCMLoss(device=device)
                dataloader_train = DataLoader(
                    train_datasets[k],
                    batch_size=conf.training.batch_size,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True,
                    num_workers=0,
                    sampler=sampler,
                )
            dataloader_val = DataLoader(
                val_datasets[k],
                batch_size=conf.training.batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=0,
            )
            # loss_func = sigmoid_focal_loss()
            # loss_func = APLoss().to(device)
            logging.info("Training for {}".format(ligand))
            model.ligand = ligand
            model.train()
            train_loss = 0.0
            all_outputs, all_labels = [], []

            for i, batch_data in tqdm(enumerate(dataloader_train)):
                index, feats_1, feats_2, labels = batch_data
                optimizer_1.zero_grad(set_to_none=True)
                # optimizer_2.zero_grad(set_to_none=True)
                optimizer_2.zero_grad()
                feats_1 = feats_1.to(device)
                feats_2 = feats_2.to(device)
                labels = labels.to(device)
                outputs = model(feats_1, feats_2)
                # loss_ = sigmoid_focal_loss(
                #     outputs * masks, labels, gamma=2, reduction="mean"
                # )
                if epoch < conf.training.warm_up_epochs:
                    loss_ = loss_func(outputs, labels)
                    loss_.backward()
                    optimizer_1.step()
                else:
                    # loss_ = loss_func(outputs, labels, index)
                    loss_ = loss_func(outputs, labels)
                    loss_.backward()
                    optimizer_2.step()
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
                all_outputs, all_labels = [], []
                for i, batch_data in tqdm(enumerate(dataloader_val)):
                    _, feats_1, feats_2, labels = batch_data
                    feats_1 = feats_1.to(device)
                    feats_2 = feats_2.to(device)
                    labels = labels.to(device)
                    outputs = model(feats_1, feats_2)
                    all_outputs.append(torch.sigmoid(outputs))
                    all_labels.append(labels)

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
