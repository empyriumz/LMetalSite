import torch
import json
import argparse
from timeit import default_timer as timer
from ml_collections import config_dict
from pathlib import Path
import datetime
import logging
import re
from script.utils import (
    logging_related,
    parse_arguments,
)
from transformers import T5EncoderModel, T5Tokenizer
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
)
from data.data_process import process_train_fasta, padding, calculate_pos_weight
from model.finetune_model import MetalIonSiteClassification


def train(conf):
    RANDOM_SEED = int(conf.general.seed)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    device = (
        torch.device("cuda:{}".format(conf.general.gpu_id))
        if torch.cuda.is_available()
        else "cpu"
    )
    batch_size = conf.training.batch_size
    log_interval = 4 * batch_size

    backbone_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_uniref50", do_lower_case=False
    )

    model = MetalIonSiteClassification(backbone_model, conf).to(device)
    if conf.training.fix_backbone_weight:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=conf.training.learning_rate,
            weight_decay=conf.training.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.classifier.parameters()},
                {
                    "params": model.backbone.parameters(),
                    "lr": conf.training.backbone_learning_rate,
                },
            ],
            lr=conf.training.learning_rate,
            weight_decay=conf.training.weight_decay,
        )

    metric_auc = BinaryAUROC(thresholds=None)
    metric_auprc = BinaryAveragePrecision(thresholds=None)
    for ion in ["CA"]:
        model.ion_type = ion
        fasta_path = conf.data.data_path + "/{}_train.fa".format(ion)
        ID_list, seq_list, label_list = process_train_fasta(
            fasta_path, conf.data.max_seq_len
        )
        pos_weight = calculate_pos_weight(label_list)
        loss_func = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.sqrt(torch.tensor(pos_weight))
        )
        #     logging.info(
        #     "Total train samples {}, val samples {}".format(len(train_data), len(val_data))
        # )
        for epoch in range(conf.training.epochs):
            train_loss, train_auc, train_auprc = 0.0, 0.0, 0.0
            model.train()
            for i, j in enumerate(range(0, len(ID_list), batch_size)):
                batch_seq_list = seq_list[j : j + batch_size]
                batch_label_list = label_list[j : j + batch_size]
                # Load sequences and map rarely occurred amino acids (U,Z,O,B) to (X)
                batch_seq_list = [
                    re.sub(r"[UZOB]", "X", " ".join(list(sequence)))
                    for sequence in batch_seq_list
                ]
                ids = tokenizer.batch_encode_plus(
                    batch_seq_list, add_special_tokens=True, padding=True
                )
                input_ids = torch.tensor(ids["input_ids"]).to(device)
                masks = torch.tensor(ids["attention_mask"]).to(device)
                labels = padding(batch_label_list, input_ids.size(1)).to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(input_ids=input_ids, attention_mask=masks)
                loss_ = loss_func(outputs * masks, labels)
                loss_.backward()
                optimizer.step()
                running_loss = loss_.detach().cpu().numpy()
                labels = torch.masked_select(labels, masks.bool())
                outputs = torch.sigmoid(torch.masked_select(outputs, masks.bool()))
                running_auc = metric_auc(outputs, labels)
                running_auprc = metric_auprc(outputs, labels)
                train_loss += running_loss
                train_auc += running_auc.detach().cpu().numpy()
                train_auprc += running_auprc.detach().cpu().numpy()

                if i % log_interval == 0:
                    logging.info("Running train loss: {:.4f}".format(running_loss))

            logging.info(
                "Epoch {} train loss {:.4f}, auc {:.3f}, auprc: {:.3f}".format(
                    epoch + 1,
                    train_loss / (i + 1),
                    train_auc / (i + 1),
                    train_auprc / (i + 1),
                )
            )
            # model.eval()
            # with torch.no_grad():
            #     val_loss = 0.0
            #     for i, j in tqdm(enumerate(range(0, len(val_data), batch_size))):
            #         partital_data = val_data[j : j + batch_size]
            #         _, _, batch_tokens = batch_converter(partital_data)
            #         sequence_rep, representation, mask = model(batch_tokens.to(device))
            #         loss_ = loss_func(sequence_rep * mask, representation * mask)
            #         val_loss += loss_.detach().cpu().numpy()

            #     validation_loss = val_loss / (i + 1)
            #     logging.info(
            #         "Epoch {} val loss: {:.4f}\n".format(epoch + 1, validation_loss)
            #     )


if __name__ == "__main__":
    start = timer()
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
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

    train(conf)
    end = timer()
    logging.info("Total time used: {:.1f}".format(end - start))
