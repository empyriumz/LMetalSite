import re
import torch
import logging
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from .utils import calculate_pos_weight, process_fasta, feature_extraction


class MetalDatasetTest(Dataset):
    def __init__(self, df, protein_features):
        self.df = df
        self.protein_features = protein_features
        self.feature_dim = 1024

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        seq_id = self.df.loc[idx, "ID"]
        protein_feat = self.protein_features[seq_id]
        return protein_feat

    def padding(self, batch, maxlen):
        batch_protein_feat = []
        batch_protein_mask = []
        for protein_feat in batch:
            padded_protein_feat = np.zeros((maxlen, self.feature_dim))
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

        return batch_protein_feat, batch_protein_mask


class MetalDataset(Dataset):
    def __init__(self, ID_list, seq_list, label_list, protein_features):
        pred_dict = {"ID": ID_list, "Sequence": seq_list, "Label": label_list}
        self.df = pd.DataFrame(pred_dict)
        self.protein_features = protein_features
        self.feature_dim = protein_features[ID_list[0]].shape[1]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        data = self.df.loc[idx]
        protein_feat = self.protein_features[data["ID"]]
        protein_label = [int(label) for label in data["Label"]]
        return protein_feat, protein_label

    def padding(self, batch, maxlen):
        batch_protein_feat = []
        batch_protein_label = []
        batch_protein_mask = []
        for protein_feat, protein_label in batch:
            padded_protein_feat = np.zeros((maxlen, self.feature_dim))
            padded_protein_feat[: protein_feat.shape[0]] = protein_feat
            padded_protein_feat = torch.tensor(padded_protein_feat, dtype=torch.float)
            batch_protein_feat.append(padded_protein_feat)

            protein_mask = np.zeros(maxlen)
            protein_mask[: protein_feat.shape[0]] = 1
            protein_mask = torch.tensor(protein_mask, dtype=torch.long)
            batch_protein_mask.append(protein_mask)

            padded_protein_label = np.zeros(maxlen)
            padded_protein_label[: protein_feat.shape[0]] = protein_label
            padded_protein_label = torch.tensor(padded_protein_label, dtype=torch.float)
            batch_protein_label.append(padded_protein_label)

        return (
            torch.stack(batch_protein_feat),
            torch.stack(batch_protein_label),
            torch.stack(batch_protein_mask),
        )

    def collate_fn(self, batch):
        maxlen = max([len(protein_label) for _, protein_label in batch])
        protein_feat, protein_label, protein_mask = self.padding(batch, maxlen)

        return protein_feat, protein_label, protein_mask, maxlen


class MultiModalDataset(Dataset):
    def __init__(self, ID_list, seq_list, label_list, protein_features):
        pred_dict = {"ID": ID_list, "Sequence": seq_list, "Label": label_list}
        self.df = pd.DataFrame(pred_dict)
        self.protein_features = protein_features
        self.feature_dim_1 = protein_features[ID_list[0]][0].shape[1]
        self.feature_dim_2 = protein_features[ID_list[0]][1].shape[1]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        data = self.df.loc[idx]
        protein_feat = self.protein_features[data["ID"]]
        feat_1, feat_2 = protein_feat[0], protein_feat[1]
        protein_label = [int(label) for label in data["Label"]]
        return feat_1, feat_2, protein_label

    def padding(self, batch, maxlen):
        batch_feat_1 = []
        batch_feat_2 = []
        batch_protein_label = []
        batch_protein_mask = []
        for feat_1, feat_2, protein_label in batch:
            padded_feat_1 = np.zeros((maxlen, self.feature_dim_1))
            padded_feat_1[: feat_1.shape[0]] = feat_1
            padded_feat_1 = torch.tensor(padded_feat_1, dtype=torch.float)
            batch_feat_1.append(padded_feat_1)

            padded_feat_2 = np.zeros((maxlen, self.feature_dim_2))
            padded_feat_2[: feat_2.shape[0]] = feat_2
            padded_feat_2 = torch.tensor(padded_feat_2, dtype=torch.float)
            batch_feat_2.append(padded_feat_2)

            protein_mask = np.zeros(maxlen)
            protein_mask[: feat_1.shape[0]] = 1
            protein_mask = torch.tensor(protein_mask, dtype=torch.long)
            batch_protein_mask.append(protein_mask)

            padded_label = np.zeros(maxlen)
            padded_label[: feat_1.shape[0]] = protein_label
            padded_label = torch.tensor(padded_label, dtype=torch.float)
            batch_protein_label.append(padded_label)

        return (
            torch.stack(batch_feat_1),
            torch.stack(batch_feat_2),
            torch.stack(batch_protein_label),
            torch.stack(batch_protein_mask),
        )

    def collate_fn(self, batch):
        maxlen = max([len(protein_label) for _, _, protein_label in batch])
        feat_1, feat_2, protein_label, protein_mask = self.padding(batch, maxlen)

        return feat_1, feat_2, protein_label, protein_mask


class FineTuneDataset(Dataset):
    def __init__(self, ID_list, seq_list, label_list, tokenizer):
        self.ID_list = ID_list
        self.seq_list = seq_list
        self.label_list = label_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ID_list)

    def padding(self, batch, max_len):
        batch_input_token = []
        batch_protein_label = []
        batch_protein_mask = []
        for input_token, protein_label in batch:
            padded_token = np.zeros(max_len)
            padded_token[: len(input_token)] = input_token
            padded_token = torch.tensor(padded_token, dtype=torch.long)
            batch_input_token.append(padded_token)

            protein_mask = np.zeros(max_len)
            protein_mask[: len(input_token)] = 1
            protein_mask = torch.tensor(protein_mask, dtype=torch.long)
            batch_protein_mask.append(protein_mask)

            padded_protein_label = np.zeros(max_len)
            padded_protein_label[: len(protein_label)] = protein_label
            padded_protein_label = torch.tensor(padded_protein_label, dtype=torch.float)
            batch_protein_label.append(padded_protein_label)

        return (
            torch.stack(batch_input_token),
            torch.stack(batch_protein_label),
            torch.stack(batch_protein_mask),
        )

    def __getitem__(self, idx):
        seq_list = re.sub(r"[UZOB]", "X", " ".join(list(self.seq_list[idx])))
        ids = self.tokenizer.encode_plus(
            seq_list, add_special_tokens=True, padding=True
        )
        labels = [int(label) for label in self.label_list[idx]]

        return ids["input_ids"], labels

    def collate_fn(self, batch):
        max_len = max([len(input_id) for input_id, _ in batch])
        input_seqs, protein_label, protein_mask = self.padding(batch, max_len)

        return input_seqs, protein_label, protein_mask


def prep_dataset(conf, device, ion_type="ZN", tokenizer=None):
    assert conf.data.data_type in ["original", "finetune", "multi_modal"]
    fasta_path = conf.data.data_path + "/{}_train.fa".format(ion_type)
    ID_list, seq_list, label_list = process_fasta(fasta_path, conf.data.max_seq_len)
    protein_features = feature_extraction(
        ID_list, seq_list, conf, device, ion_type=ion_type, model_name=conf.model.name
    )
    pos_weight = calculate_pos_weight(label_list)
    if conf.data.data_type == "original":
        dataset = MetalDataset(ID_list, seq_list, label_list, protein_features)
    elif conf.data.data_type == "finetune":
        assert tokenizer is not None, "Invalid tokenizer"
        dataset = FineTuneDataset(ID_list, seq_list, label_list, tokenizer)
    elif conf.data.data_type == "multi_modal":
        dataset = MultiModalDataset(ID_list, seq_list, label_list, protein_features)

    return dataset, pos_weight


def prep_dataloader(dataset, conf, random_seed=0, ion_type="ZN"):
    n_val = int(len(dataset) * conf.training.val_ratio)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(random_seed)
    )
    logging.info(
        "\nTraining sequences for {}: {}; validation sequences {}".format(
            ion_type, len(train_dataset), len(val_dataset)
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

    return train_dataloader, val_dataloader


def finetune_data_loader(conf, tokenizer, random_seed=0, ion_type="ZN"):
    fasta_path = conf.data.data_path + "/{}_train.fa".format(ion_type)
    ID_list, seq_list, label_list = process_fasta(fasta_path, conf.data.max_seq_len)
    pos_weight = calculate_pos_weight(label_list)
    dataset = FineTuneDataset(ID_list, seq_list, label_list, tokenizer)
    n_val = int(len(dataset) * conf.training.val_ratio)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(random_seed)
    )
    logging.info(
        "\nTraining sequences for {}: {}; validation sequences {}".format(
            ion_type, len(train_dataset), len(val_dataset)
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
        drop_last=False,
        pin_memory=True,
        num_workers=8,
    )

    return train_dataloader, val_dataloader, pos_weight
