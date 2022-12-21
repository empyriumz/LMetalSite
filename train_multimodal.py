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
    BinaryF1Score,
)
from pathlib import Path
from script.utils import (
    add_alphafold_args,
    logging_related,
    parse_arguments,
)
from data.data_process import prep_dataset, prep_dataloader
from model.model import LMetalSiteMultiModal


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
    if conf.model.name == "Evoformer":
        conf.model.feature_dim = 384
    elif conf.model.name == "ProtTrans":
        conf.model.feature_dim = 1024
    elif conf.model.name == "Composite":
        conf.model.feature_dim = 1408
    # Load LMetalSite model
    model = LMetalSiteMultiModal(conf).to(device)

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
    log_interval = 2 * conf.training.batch_size
    model.training = True  # adding Gaussian noise to embedding
    for ion in ["ZN", "MN", "MG", "CA"]:
        dataset, pos_weight = prep_dataset(conf, device, ion_type=ion)
        train_dataloader, val_dataloader = prep_dataloader(
            dataset, conf, random_seed=RANDOM_SEED, ion_type=ion
        )
        loss_func = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.sqrt(torch.tensor(pos_weight))
        )
        # loss_func = torch.nn.BCEWithLogitsLoss()
        model.ion_type = ion
        for epoch in range(conf.training.epochs):
            train_loss = 0.0
            all_outputs, all_labels = [], []
            for i, batch_data in tqdm(enumerate(train_dataloader)):
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
                running_auc = metric_auc(outputs, labels)
                running_auprc = metric_auprc(outputs, labels)
                running_loss = loss_.detach().cpu().numpy()
                train_loss += running_loss
                if i % log_interval == 0 and i > 0:
                    logging.info(
                        "Running train loss: {:.4f}, auc: {:.3f}, auprc: {:.3f}".format(
                            running_loss, running_auc, running_auprc
                        )
                    )
            all_outputs = torch.cat(all_outputs)
            all_labels = torch.cat(all_labels)
            train_auc = metric_auc(all_outputs, all_labels).detach().cpu().numpy()
            train_auprc = metric_auprc(all_outputs, all_labels).detach().cpu().numpy()
            logging.info(
                "Epoch {} train loss {:.4f}, auc {:.3f}, auprc: {:.3f}".format(
                    epoch + 1,
                    train_loss / (i + 1),
                    train_auc,
                    train_auprc,
                )
            )
            model.eval()
            with torch.no_grad():
                model.training = False
                val_loss = 0.0
                all_outputs, all_labels = [], []
                for i, batch_data in tqdm(enumerate(val_dataloader)):
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

            all_outputs = torch.cat(all_outputs)
            all_labels = torch.cat(all_labels)
            val_auc = metric_auc(all_outputs, all_labels).detach().cpu().numpy()
            val_auprc = metric_auprc(all_outputs, all_labels).detach().cpu().numpy()
            logging.info(
                "Epoch {} val loss {:.4f}, auc {:.3f}, auprc: {:.3f}".format(
                    epoch + 1,
                    val_loss / (i + 1),
                    val_auc,
                    val_auprc,
                )
            )

    logging.info(
        "Training is done at {}".format(datetime.datetime.now().strftime("%m-%d %H:%M"))
    )


if __name__ == "__main__":
    start = timer()
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    add_alphafold_args(parser)
    output_path = (
        Path("./results/")
        / Path(args.config).stem
        / Path(str(datetime.datetime.now())[:16].replace(" ", "-").replace(":", "-"))
    )
    output_path.mkdir(parents=True, exist_ok=True)

    """
    Read configuration and dump the configuration to output dir
    """
    with open(args.config, "r") as f:
        conf = json.load(f)
    conf["output_path"] = "./" + str(output_path)
    with open(str(output_path) + "/config.json", "w") as f:
        json.dump(conf, f, indent=4)

    conf = config_dict.ConfigDict(conf)
    """
    logging related part
    """
    logging_related(output_path)
    main(conf)

    end = timer()
    logging.info("Total time used: {:.1f}".format(end - start))
