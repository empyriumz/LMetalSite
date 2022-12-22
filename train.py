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
from script.utils import (
    logging_related,
    parse_arguments,
)
from data.data_process import prep_dataset, prep_dataloader
from model.model import LMetalSite, LMetalSiteBase

LOG_INTERVAL = 50


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
    if conf.data.feature == "Evoformer":
        conf.model.feature_dim = 384
    elif conf.data.feature == "ProtTrans":
        conf.model.feature_dim = 1024
    elif conf.data.feature == "Composite":
        conf.model.feature_dim = 1408
    else:
        raise ValueError("No feature available")

    # Load LMetalSite model
    if conf.model.name == "base":
        model = LMetalSiteBase(conf.model).to(device)
    elif conf.model.name == "transformer":
        model = LMetalSite(conf.model).to(device)
    else:
        raise NotImplementedError("Invalid model name")

    if conf.training.pretrained_encoder:
        checkpoint = torch.load(conf.training.pretrained_encoder)
        logging.info("load encoder from {}".format(conf.training.pretrained_encoder))
        model.input_block.load_state_dict(checkpoint["encoder_state"])

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
        optimizer = torch.optim.AdamW(
            [
                {"params": model.params.classifier.parameters()},
                {
                    "params": model.params.encoder.parameters(),
                    "lr": conf.training.encoder_learning_rate,
                },
            ],
            lr=conf.training.learning_rate,
            weight_decay=conf.training.weight_decay,
        )

    metric_auc = BinaryAUROC(thresholds=None)
    metric_auprc = BinaryAveragePrecision(thresholds=None)
    model.training = True  # adding Gaussian noise to embedding
    for ion in ["ZN", "MN", "MG", "CA"]:
        dataset, pos_weight = prep_dataset(conf, device, ion_type=ion)
        train_dataloader, val_dataloader = prep_dataloader(
            dataset, conf, random_seed=RANDOM_SEED, ion_type=ion
        )
        loss_func = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.sqrt(torch.tensor(pos_weight))
        )
        model.ion_type = ion
        for epoch in range(conf.training.epochs):
            train_loss = 0.0
            all_outputs, all_labels = [], []
            for i, batch_data in tqdm(enumerate(train_dataloader)):
                feats, labels, masks, _ = batch_data
                optimizer.zero_grad(set_to_none=True)
                feats = feats.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                outputs = model(feats, masks)
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
                if i % LOG_INTERVAL == 0 and i > 0:
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
                    feats, labels, masks, _ = batch_data
                    feats = feats.to(device)
                    masks = masks.to(device)
                    labels = labels.to(device)
                    outputs = model(feats, masks)
                    labels = torch.masked_select(labels, masks.bool())
                    outputs = torch.sigmoid(torch.masked_select(outputs, masks.bool()))
                    all_outputs.append(outputs)
                    all_labels.append(labels)
                    running_loss = loss_.detach().cpu().numpy()
                    val_loss += running_loss

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
    with open(args.config, "r") as f:
        conf = json.load(f)

    output_path = None
    if not conf["general"]["debug"]:
        output_path = (
            Path("./results/")
            / Path(args.config).stem
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
    main(conf)

    end = timer()
    logging.info("Total time used: {:.1f}".format(end - start))
