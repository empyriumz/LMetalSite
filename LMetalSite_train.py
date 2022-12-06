import os, datetime, argparse, re
import numpy as np
import pandas as pd
import json
import torch
import gc
import logging
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
from ml_collections import config_dict
from timeit import default_timer as timer
from pathlib import Path
from script.utils import (
    add_alphafold_args,
    logging_related,
    parse_arguments,
)
from data.data_process import process_train_fasta, MetalDataset
from model.model import LMetalSite

############ Set to your own path! ############
# ProtTrans_path = "/home/yuanqm/protein_binding_sites/tools/Prot-T5-XL-U50"

Max_repr = np.load("script/ProtTrans_repr_max.npy")
Min_repr = np.load("script/ProtTrans_repr_min.npy")

metal_list = ["ZN", "CA", "MG", "MN"]
LMetalSite_threshold = {"ZN": 0.42, "CA": 0.34, "MG": 0.5, "MN": 0.47}


def feature_extraction(ID_list, seq_list, conf, device):
    protein_features = {}
    if conf.data.save_feature:
        feat_path = conf.output_path + "ProtTrans_repr"
        os.makedirs(feat_path, exist_ok=True)

    # Load the vocabulary and ProtT5-XL-UniRef50 Model
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_uniref50", do_lower_case=False
    )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    gc.collect()

    # Load the model into CPU/GPU and switch to inference mode
    model = model.to(device)
    model = model.eval()
    batch_size = conf.training.feature_batch_size
    # Extract feature of one batch each time
    for i in tqdm(range(0, len(ID_list), batch_size)):
        if i + batch_size <= len(ID_list):
            batch_ID_list = ID_list[i : i + batch_size]
            batch_seq_list = seq_list[i : i + batch_size]
        else:
            batch_ID_list = ID_list[i:]
            batch_seq_list = seq_list[i:]

        # Load sequences and map rarely occured amino acids (U,Z,O,B) to (X)
        batch_seq_list = [
            re.sub(r"[UZOB]", "X", " ".join(list(sequence)))
            for sequence in batch_seq_list
        ]

        # Tokenize, encode sequences and load it into GPU if avilabile
        ids = tokenizer.batch_encode_plus(
            batch_seq_list, add_special_tokens=True, padding=True
        )
        input_ids = torch.tensor(ids["input_ids"]).to(device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(device)

        # Extract sequence features and load it into CPU if needed
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.detach().cpu().numpy()

        # Remove padding (\<pad>) and special tokens (\</s>) that is added by ProtT5-XL-UniRef50 model
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][: seq_len - 1]
            if conf.data.save_feature:
                np.save(feat_path + "/" + batch_ID_list[seq_num], seq_emd)
            # Normalization
            seq_emd = (seq_emd - Min_repr) / (Max_repr - Min_repr)
            protein_features[batch_ID_list[seq_num]] = seq_emd

    return protein_features


def main(conf):
    RANDOM_SEED = int(conf.general.seed)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    ID_list, seq_list, label_list = process_train_fasta(
        conf.data.fasta_path, conf.data.max_seq_len
    )
    device = (
        torch.device("cuda:{}".format(conf.general.gpu_id))
        if torch.cuda.is_available()
        else "cpu"
    )
    logging.info(
        "Feature extraction begins at {}".format(
            datetime.datetime.now().strftime("%m-%d %H:%M")
        )
    )

    protein_features = feature_extraction(ID_list, seq_list, conf, device)

    logging.info(
        "Feature extraction is done at {}".format(
            datetime.datetime.now().strftime("%m-%d %H:%M")
        )
    )
    logging.info(
        "raining begins at {}".format(datetime.datetime.now().strftime("%m-%d %H:%M"))
    )
    pred_dict = {"ID": ID_list, "Sequence": seq_list, "Label": label_list}
    pred_df = pd.DataFrame(pred_dict)

    ion_type = conf.model.ion_type
    pred_df[ion_type + "_prob"] = 0.0
    pred_df[ion_type + "_pred"] = 0.0

    train_dataset = MetalDataset(pred_df, protein_features, conf.model.feature_dim)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=conf.training.batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        drop_last=False,
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
    loss_func = torch.nn.BCEWithLogitsLoss()

    # Make Predictions
    prob_col, pred_col = [], []
    for epoch in range(conf.training.epochs):
        train_loss = 0.0
        for batch_data in tqdm(train_dataloader):
            protein_feats, protein_labels, protein_masks, maxlen = batch_data
            optimizer.zero_grad(set_to_none=True)
            protein_feats = protein_feats.to(device)
            protein_masks = protein_masks.to(device)
            protein_labels = protein_labels.to(device)
            outputs = model(protein_feats, protein_masks)
            loss_ = loss_func(outputs, protein_labels)
            loss_.backward()
            optimizer.step()
            outputs = (
                outputs.detach().cpu().numpy()
            )  # shape = (pred_bs, len(metal_list) * maxlen)
            running_loss = loss_.detach().cpu().numpy()
            train_loss += running_loss
        #     for j in range(len(outputs)):
        #         prob = np.round(
        #             outputs[j, i * maxlen : i * maxlen + protein_masks[j].sum()],
        #             decimals=4,
        #         )
        #         pred = (prob >= LMetalSite_threshold[ion_type]).astype(int)
        #         prob_col.append(",".join(prob.astype(str).tolist()))
        #         pred_col.append(",".join(pred.astype(str).tolist()))

        # pred_df[ion_type + "_prob"] = prob_col
        # pred_df[ion_type + "_pred"] = pred_col

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
