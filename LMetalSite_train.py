import datetime
import argparse
import pandas as pd
import json
import torch
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
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
from data.data_process import (
    process_train_fasta,
    feature_extraction,
    calculate_pos_weight,
    MetalDataset,
)
from model.model import LMetalSite


def main(conf):
    RANDOM_SEED = int(conf.general.seed)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    device = (
        torch.device("cuda:{}".format(conf.general.gpu_id))
        if torch.cuda.is_available()
        else "cpu"
    )
    ID_list, seq_list, label_list = process_train_fasta(
        conf.data.fasta_path, conf.data.max_seq_len
    )

    logging.info(
        "Feature extraction begins at {}".format(
            datetime.datetime.now().strftime("%m-%d %H:%M")
        )
    )
    protein_features = feature_extraction(
        ID_list, seq_list, conf, device, model_name=conf.model.name
    )
    logging.info(
        "Feature extraction is done at {}".format(
            datetime.datetime.now().strftime("%m-%d %H:%M")
        )
    )
    logging.info(
        "Training begins at {}".format(datetime.datetime.now().strftime("%m-%d %H:%M"))
    )
    pred_dict = {"ID": ID_list, "Sequence": seq_list, "Label": label_list}
    pred_df = pd.DataFrame(pred_dict)

    dataset = MetalDataset(pred_df, protein_features, conf.model.feature_dim)
    n_val = int(len(dataset) * conf.training.val_ratio)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    logging.info(
        "Total training sequences {}; validation sequences {}".format(
            len(train_dataset), len(val_dataset)
        )
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=conf.training.batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=8,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=conf.training.batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=8,
    )
    # Load LMetalSite model
    model = LMetalSite(
        conf.model.feature_dim,
        conf.model.hidden_dim,
        conf.model.num_encoder_layers,
        conf.model.num_heads,
        conf.model.augment_eps,
        conf.model.dropout,
        conf.model.ion_type,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=conf.training.learning_rate,
        weight_decay=conf.training.weight_decay,
    )
    # loss_func = torch.nn.BCEWithLogitsLoss()
    pos_weight = calculate_pos_weight(label_list)
    loss_func = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.sqrt(torch.tensor(pos_weight))
    )
    metric_auc = BinaryAUROC(thresholds=None)
    metric_f1 = BinaryF1Score()
    metric_auprc = BinaryAveragePrecision(thresholds=None)
    log_interval = 2 * conf.training.batch_size
    model.training = True  # adding Gaussian noise to embedding
    for epoch in range(conf.training.epochs):
        train_loss, train_auc, train_f1, train_auprc = 0.0, 0.0, 0.0, 0.0
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
            running_auc = metric_auc(outputs, labels)
            running_auprc = metric_auprc(outputs, labels)
            # running_f1 = metric_f1(outputs, labels)
            running_loss = loss_.detach().cpu().numpy()
            train_loss += running_loss
            train_auc += running_auc.detach().cpu().numpy()
            train_auprc += running_auprc.detach().cpu().numpy()
            # train_f1 += running_f1.detach().cpu().numpy()
            if i % log_interval == 0 and i > 0:
                logging.info(
                    "Running train loss: {:.4f}, auc: {:.3f}, auprc: {:.3f}".format(
                        running_loss, running_auc, running_auprc
                    )
                )
        logging.info(
            "Epoch {} train loss {:.4f}, auc {:.3f}, auprc: {:.3f}".format(
                epoch + 1,
                train_loss / (i + 1),
                train_auc / (i + 1),
                train_auprc / (i + 1),
            )
        )
        model.eval()
        with torch.no_grad():
            model.training = False
            val_loss, val_auc, val_f1, val_auprc = 0.0, 0.0, 0.0, 0.0
            for i, batch_data in tqdm(enumerate(val_dataloader)):
                feats, labels, masks, _ = batch_data
                feats = feats.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                outputs = model(feats, masks)
                labels = torch.masked_select(labels, masks.bool())
                outputs = torch.sigmoid(torch.masked_select(outputs, masks.bool()))
                running_auc = metric_auc(outputs, labels).detach().cpu().numpy()
                running_auprc = metric_auprc(outputs, labels).detach().cpu().numpy()
                # running_f1 = metric_f1(outputs, labels).detach().cpu().numpy()
                running_loss = loss_.detach().cpu().numpy()
                val_loss += running_loss
                val_auc += running_auc
                val_auprc += running_auprc
                # val_f1 += running_f1

        logging.info(
            "Epoch {} val loss {:.4f}, auc {:.3f}, auprc: {:.3f}".format(
                epoch + 1,
                val_loss / (i + 1),
                val_auc / (i + 1),
                val_auprc / (i + 1),
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
