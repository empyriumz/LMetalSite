import string, re
import numpy as np
import torch
import argparse
from datetime import date
import logging
import sys

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


class MetalDataset:
    def __init__(self, df, protein_features):
        self.df = df
        self.protein_features = protein_features
        self.feat_dim = NN_config["feature_dim"]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        seq_id = self.df.loc[idx, ID_col]
        protein_feat = self.protein_features[seq_id]
        return protein_feat

    def padding(self, batch, maxlen):
        batch_protein_feat = []
        batch_protein_mask = []
        for protein_feat in batch:
            padded_protein_feat = np.zeros((maxlen, self.feat_dim))
            padded_protein_feat[: protein_feat.shape[0]] = protein_feat
            padded_protein_feat = torch.tensor(padded_protein_feat, dtype=torch.float)
            batch_protein_feat.append(padded_protein_feat)

            protein_mask = np.zeros(maxlen)
            protein_mask[: protein_feat.shape[0]] = 1
            protein_mask = torch.tensor(protein_mask, dtype=torch.long)
            batch_protein_mask.append(protein_mask)

        return torch.stack(batch_protein_feat), torch.stack(batch_protein_mask)

    def collate_fn(self, batch):
        maxlen = max([protein_feat.shape[0] for protein_feat in batch])
        batch_protein_feat, batch_protein_mask = self.padding(batch, maxlen)

        return batch_protein_feat, batch_protein_mask, maxlen


def process_fasta(fasta_file, max_input_seq_num):
    ID_list = []
    seq_list = []

    with open(fasta_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line[0] == ">":
            name_item = line[1:-1].split("|")
            ID = "_".join(name_item[0 : min(2, len(name_item))])
            ID = re.sub(" ", "_", ID)
            ID_list.append(ID)
        elif line[0] in string.ascii_letters:
            seq_list.append(line.strip().upper())

    if len(ID_list) == len(seq_list):
        if len(ID_list) > max_input_seq_num:
            raise ValueError("Too much sequences! Up to {} sequences are supported each time!".format(
                max_input_seq_num
            ))
        else:
            return [ID_list, seq_list]
    else:
        raise ValueError("The format of input fasta file is incorrect")

def parse_arguments(parser):
    parser.add_argument(
        "--config", type=str, default=None, help="Name of configuration file"
    )
    parser.add_argument("--fasta", type=str, help="Input fasta file")
    parser.add_argument(
        "--outpath",
        type=str,
        help="Output path to save intermediate features and final predictions",
    )
    parser.add_argument(
        "--feat_bs",
        type=int,
        default=8,
        help="Batch size for ProtTrans feature extraction",
    )
    parser.add_argument(
        "--pred_bs", type=int, default=16, help="Batch size for LMetalSite prediction"
    )
    parser.add_argument(
        "--save_feat", action="store_true", help="Save intermediate ProtTrans features"
    )
    parser.add_argument(
        "--gpu",
        default=1,
        action="store_true",
        help="Use GPU for feature extraction and LMetalSite prediction",
    )
    args = parser.parse_args()
    return args


def add_alphafold_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--uniref90_database_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--mgnify_database_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pdb70_database_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--uniclust30_database_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--bfd_database_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--jackhmmer_binary_path", type=str, default="/usr/bin/jackhmmer"
    )
    parser.add_argument(
        "--hhblits_binary_path",
        type=str,
        default="/opt/conda/envs/openfold/bin/hhblits",
    )
    parser.add_argument(
        "--hhsearch_binary_path",
        type=str,
        default="/opt/conda/envs/openfold/bin/hhsearch",
    )
    parser.add_argument(
        "--kalign_binary_path", type=str, default="/opt/conda/envs/openfold/bin/kalign"
    )
    parser.add_argument(
        "--max_template_date",
        type=str,
        default=date.today().strftime("%Y-%m-%d"),
    )
    parser.add_argument("--obsolete_pdbs_path", type=str, default=None)
    parser.add_argument("--release_dates_path", type=str, default=None)

def logging_related(output_path):
    logger = logging.getLogger()
    log_filename = str(output_path) + "/training.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logging.info("Output path: {}".format(output_path))
