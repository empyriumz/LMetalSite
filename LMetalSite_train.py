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
    MetalDataset,
    process_fasta,
    add_alphafold_args,
    logging_related,
    parse_arguments,
)
from model.model import LMetalSite

############ Set to your own path! ############
# ProtTrans_path = "/home/yuanqm/protein_binding_sites/tools/Prot-T5-XL-U50"

script_path = os.path.split(os.path.realpath(__file__))[0] + "/"
model_path = os.path.dirname(script_path[0:-1]) + "/model/"

Max_repr = np.load(script_path + "ProtTrans_repr_max.npy")
Min_repr = np.load(script_path + "ProtTrans_repr_min.npy")

MAX_INPUT_SEQ = 500
ID_col = "ID"
sequence_col = "Sequence"
metal_list = ["ZN", "CA", "MG", "MN"]
LMetalSite_threshold = [0.42, 0.34, 0.5, 0.47]

NN_config = {
    "feature_dim": 1024,
    "hidden_dim": 64,
    "num_encoder_layers": 2,
    "num_heads": 4,
    "augment_eps": 0.05,
    "dropout": 0.2,
}


def feature_extraction(ID_list, seq_list, conf, device):
    protein_features = {}
    if conf.data.save_feat:
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
    batch_size = conf.training.feat_batch_size
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
            if conf.data.save_feat:
                np.save(feat_path + "/" + batch_ID_list[seq_num], seq_emd)
            # Normalization
            seq_emd = (seq_emd - Min_repr) / (Max_repr - Min_repr)
            protein_features[batch_ID_list[seq_num]] = seq_emd

    return protein_features


def train(ID_list, seq_list, protein_features, conf, device):
    RANDOM_SEED = int(conf.general.seed)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    pred_dict = {ID_col: ID_list, sequence_col: seq_list}
    pred_df = pd.DataFrame(pred_dict)

    for metal in metal_list:
        pred_df[metal + "_prob"] = 0.0
        pred_df[metal + "_pred"] = 0.0

    train_dataset = MetalDataset(pred_df, protein_features)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=conf.training.batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )
    if conf.training.fix_backbone_weight:
        optimizer = torch.optim.Adam(
            model.classifier.parameters(),
            lr=conf.training.learning_rate,
            weight_decay=conf.training.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            [
                {"params": model.params.classifier.parameters()},
                {
                    "params": model.params.backbone.parameters(),
                    "lr": conf.training.backbone_learning_rate,
                },
            ],
            lr=conf.training.learning_rate,
            weight_decay=conf.training.weight_decay,
        )
    # Load LMetalSite model
    model = LMetalSite(
        conf.training.feature_dim,
        conf.training.hidden_dim,
        conf.training.num_encoder_layers,
        conf.training.num_heads,
        conf.training.augment_eps,
        conf.training.dropout,
    ).to(device)

    # Make Predictions
    prob_col = [[] for _ in range(len(metal_list))]
    pred_col = [[] for _ in range(len(metal_list))]
    for epoch in range(conf.training.epochs):
        for batch_data in tqdm(train_dataloader):
            protein_feats, protein_masks, maxlen = batch_data
            protein_feats = protein_feats.to(device)
            protein_masks = protein_masks.to(device)

            outputs = model(protein_feats, protein_masks).sigmoid()

            outputs = (
                outputs.detach().cpu().numpy()
            )  # shape = (pred_bs, len(metal_list) * maxlen)

            for i in range(len(metal_list)):
                for j in range(len(outputs)):
                    prob = np.round(
                        outputs[j, i * maxlen : i * maxlen + protein_masks[j].sum()],
                        decimals=4,
                    )
                    pred = (prob >= LMetalSite_threshold[i]).astype(int)
                    prob_col[i].append(",".join(prob.astype(str).tolist()))
                    pred_col[i].append(",".join(pred.astype(str).tolist()))

        for i in range(len(metal_list)):
            pred_df[metal_list[i] + "_prob"] = prob_col[i]
            pred_df[metal_list[i] + "_pred"] = pred_col[i]

    # pred_df.to_csv(conf.output_path + "_predictions.csv", index=False)


def main(conf):
    ID_list, seq_list = process_fasta(conf.fasta, conf.max_seq_len)
    device = (
        torch.device("cuda:{}".format(conf.general.gpu_id))
        if torch.cuda.is_available()
        else "cpu"
    )
    logging.info(
        "\n######## Feature extraction begins at {}. ########\n".format(
            datetime.datetime.now().strftime("%m-%d %H:%M")
        )
    )

    protein_features = feature_extraction(ID_list, seq_list, conf, device)

    logging.info(
        "\n######## Feature extraction is done at {}. ########\n".format(
            datetime.datetime.now().strftime("%m-%d %H:%M")
        )
    )

    logging.info(
        "\n######## Training begins at {}. ########\n".format(
            datetime.datetime.now().strftime("%m-%d %H:%M")
        )
    )

    train(ID_list, seq_list, protein_features, device)

    logging.info(
        "\n######## Training is done at {}. ########\n".format(
            datetime.datetime.now().strftime("%m-%d %H:%M")
        )
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
