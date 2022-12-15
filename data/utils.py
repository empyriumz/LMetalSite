import numpy as np
import torch
import re
import string


def process_fasta(fasta_file, max_input_seq_num=5000):
    ID_list = []
    seq_list = []
    label_list = []

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
        elif line[0] in string.digits:
            label_list.append(line.strip())

    if len(ID_list) == len(seq_list):
        if len(ID_list) > max_input_seq_num:
            raise ValueError(
                "Too much sequences! Up to {} sequences are supported each time!".format(
                    max_input_seq_num
                )
            )
        else:
            if len(label_list) > 0:
                return [ID_list, seq_list, label_list]
            else:
                return [ID_list, seq_list]
    else:
        raise ValueError("The format of input fasta file is incorrect")


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
