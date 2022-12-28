import numpy as np
import torch
from esm import pretrained


def load_esm_model(backend_name):
    if backend_name == "esm_tiny":  # embed_dim=320
        backbone_model, alphabet = pretrained.esm2_t6_8M_UR50D()
    elif backend_name == "esm_small":  # embed_dim=640
        backbone_model, alphabet = pretrained.esm2_t30_150M_UR50D()
    elif backend_name == "esm_medium":  # embed_dim=1280
        backbone_model, alphabet = pretrained.esm2_t33_650M_UR50D()
    elif backend_name == "esm_large":  # embed_dim=1280
        backbone_model, alphabet = pretrained.esm2_t36_3B_UR50D()
    elif backend_name == "esm_1b":  # embed_dim=1280
        backbone_model, alphabet = pretrained.esm1b_t33_650M_UR50S()
        backbone_model.embed_dim = 1280
    else:
        raise ValueError("Wrong backbone model")

    return backbone_model, alphabet


def padding(batch, maxlen):
    batch_list = []
    for label in batch:
        protein_label = [int(i) for i in label]
        padded_list = np.zeros(maxlen)
        padded_list[: len(protein_label)] = protein_label
        padded_list = torch.tensor(padded_list, dtype=torch.float)
        batch_list.append(padded_list)

    return torch.stack(batch_list)


def calculate_pos_weight(label_list):
    pos_num, neg_num = [], []
    for label in label_list:
        pos_ = sum([int(i) for i in label])
        pos_num.append(pos_)
        neg_num.append(len(label) - pos_)

    pos_weight = sum(neg_num) / sum(pos_num)
    return pos_weight
