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
from data.data_process import prep_dataset, prep_dataloader
from model.model import LMetalSiteMultiModal, LMetalSiteMultiModalBase


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

    if conf.training.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            betas=(0.9, 0.99),
            lr=conf.training.learning_rate,
            weight_decay=conf.training.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=conf.training.learning_rate,
            weight_decay=conf.training.weight_decay,
        )
    metric_auc = BinaryAUROC(thresholds=None)
    metric_auprc = BinaryAveragePrecision(thresholds=None)
    model.training = True  # adding Gaussian noise to embedding
    ligand_list = ["DNA", "RNA", "MG", "CA", "MN", "ZN"]
    pos_weights = []
    dataloader_train, dataloader_val = [], []
    for ligand in ligand_list:
        dataset, pos_weight = prep_dataset(conf, device, ligand=ligand)
        train_dataloader, val_dataloader = prep_dataloader(
            dataset, conf, random_seed=RANDOM_SEED, ligand=ligand
        )
        dataloader_train.append(train_dataloader)
        dataloader_val.append(val_dataloader)
        pos_weights.append(pos_weight)

    for epoch in range(conf.training.epochs):
        for k, ligand in enumerate(ligand_list):
            loss_func = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.sqrt(torch.tensor(pos_weights[k]))
            )
            logging.info("Training for {}".format(ligand))
            model.ligand = ligand
            model.train()
            train_loss = 0.0
            all_outputs, all_labels = [], []
            for i, batch_data in tqdm(enumerate(dataloader_train[k])):
                feats_1, feats_2, labels, masks = batch_data
                optimizer.zero_grad(set_to_none=True)
                feats_1 = feats_1.to(device)
                feats_2 = feats_2.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                outputs = model(feats_1, feats_2)
                loss_ = loss_func(outputs * masks, labels)
                loss_.backward()
                optimizer.step()
                labels = torch.masked_select(labels, masks.bool())
                outputs = torch.sigmoid(torch.masked_select(outputs, masks.bool()))
                all_outputs.append(outputs)
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
                    feats_1, feats_2, labels, masks = batch_data
                    feats_1 = feats_1.to(device)
                    feats_2 = feats_2.to(device)
                    masks = masks.to(device)
                    labels = labels.to(device)
                    outputs = model(feats_1, feats_2)
                    labels = torch.masked_select(labels, masks.bool())
                    outputs = torch.sigmoid(torch.masked_select(outputs, masks.bool()))
                    all_outputs.append(outputs)
                    all_labels.append(labels)
                    val_loss += loss_.detach().cpu().numpy()

                all_outputs = torch.cat(all_outputs).detach().cpu()
                all_labels = torch.cat(all_labels).detach().cpu()
                val_auc = metric_auc(all_outputs, all_labels)
                val_auprc = metric_auprc(all_outputs, all_labels)
                logging.info(
                    "Epoch {} val loss {:.4f}, auc {:.3f}, auprc: {:.3f}".format(
                        epoch + 1,
                        val_loss / (i + 1),
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
