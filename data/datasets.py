import torch
import re
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


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
    def __init__(self, ID_list, label_list, protein_features):
        pred_dict = {"ID": ID_list, "Label": label_list}
        self.df = pd.DataFrame(pred_dict)
        self.protein_features = protein_features
        self.feature_dim = protein_features[ID_list[0]].shape[1]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        data = self.df.loc[idx]
        protein_feat = self.protein_features[data["ID"]]
        protein_label = data["Label"]
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

        return protein_feat, protein_label, protein_mask


class UniRefDataset(Dataset):
    def __init__(self, ID_list, protein_features):
        pred_dict = {"ID": ID_list}
        self.df = pd.DataFrame(pred_dict)
        self.protein_features = protein_features
        self.feature_dim = protein_features[ID_list[0]].shape[1]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        data = self.df.loc[idx]
        protein_feat = self.protein_features[data["ID"]]
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

        return (
            torch.stack(batch_protein_feat),
            torch.stack(batch_protein_mask),
        )

    def collate_fn(self, batch):
        maxlen = max([len(feat) for feat in batch])
        protein_feat, protein_mask = self.padding(batch, maxlen)

        return protein_feat, protein_mask


class MultiModalDataset(Dataset):
    def __init__(self, ID_list, label_data, protein_features, mode=None):
        self.mode = mode
        feat_1 = []
        feat_2 = []
        label_list = []
        for id in ID_list:
            feat_1.append(protein_features[id][0])
            feat_2.append(protein_features[id][1])
            label_list.append(label_data[id])
        self.targets = np.concatenate(label_list, dtype=np.float32)
        self.feat_1 = np.concatenate(feat_1)
        self.feat_2 = np.concatenate(feat_2)
        # for loss function
        self.pos_indices = np.flatnonzero(self.targets == 1)
        self.pos_index_map = {}
        for i, idx in enumerate(self.pos_indices):
            self.pos_index_map[idx] = i

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        feat_1 = self.feat_1[idx]
        feat_2 = self.feat_2[idx]
        target = self.targets[idx]
        if self.mode == "train":
            pos_idx = self.pos_index_map[idx] if idx in self.pos_indices else -1
        else:
            pos_idx = idx
        return pos_idx, feat_1, feat_2, target


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
