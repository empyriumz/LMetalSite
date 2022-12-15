import string
import re
import torch
import os
import gc
import logging
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
try:
    from transformers import T5EncoderModel, T5Tokenizer
except:
    pass


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

        return batch_protein_feat, batch_protein_mask, maxlen


class MetalDataset(Dataset):
    def __init__(self, df, protein_features, feature_dim):
        self.df = df
        self.protein_features = protein_features
        self.feature_dim = feature_dim

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


def prottrans_embedding(ID_list, seq_list, conf, device, normalize=True, ion_type="ZN"):
    protein_features = {}
    max_repr = np.load("script/ProtTrans_repr_max.npy")
    min_repr = np.load("script/ProtTrans_repr_min.npy")
    if conf.data.precomputed_feature:
        for id in ID_list:
            seq_emd = np.load(
                conf.data.precomputed_feature
                + "/{}_ProtTrans/{}.npy".format(ion_type, id)
            )
            if normalize:
                seq_emd = (seq_emd - min_repr) / (max_repr - min_repr)
            protein_features[id] = seq_emd

        return protein_features

    if conf.data.save_feature:
        feat_path = conf.output_path + "/{}_ProtTrans".format(ion_type)
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
            seq_emd = (seq_emd - min_repr) / (max_repr - min_repr)
            protein_features[batch_ID_list[seq_num]] = seq_emd

    return protein_features


def load_evoformer_embedding(
    ID_list, precomputed_feature_path, normalize=True, ion_type="ZN"
):
    protein_features = {}
    max_repr = np.load("script/Evoformer_repr_max.npy")
    min_repr = np.load("script/Evoformer_repr_min.npy")
    # max_repr = np.load("script/Evoformer_pair_repr_max.npy")
    # min_repr = np.load("script/Evoformer_pair_repr_min.npy")
    for id in ID_list:
        feature = np.load(
            precomputed_feature_path + "/{}_Evoformer/{}.npz".format(ion_type, id)
        )
        # seq_emd = np.concatenate((feature["single"], feature["pair"]), axis=1)
        seq_emd = feature["single"]
        # seq_emd = feature["pair"]
        if normalize:
            seq_emd = (seq_emd - min_repr) / (max_repr - min_repr)
        protein_features[id] = seq_emd

    return protein_features


def composite_embedding(
    ID_list, precomputed_feature_path, normalize=True, ion_type="ZN"
):
    max_repr_evo = np.load("script/Evoformer_repr_max.npy")
    min_repr_evo = np.load("script/Evoformer_repr_min.npy")
    max_repr_prot = np.load("script/ProtTrans_repr_max.npy")
    min_repr_prot = np.load("script/ProtTrans_repr_min.npy")
    min_repr = np.concatenate((min_repr_prot, min_repr_evo))
    max_repr = np.concatenate((max_repr_prot, max_repr_evo))
    protein_features = {}
    for id in ID_list:
        feature_evo = np.load(
            precomputed_feature_path + "/{}_Evoformer/{}.npz".format(ion_type, id)
        )
        feature_prot = np.load(
            precomputed_feature_path + "/{}_ProtTrans/{}.npy".format(ion_type, id)
        )
        seq_emd = np.concatenate((feature_prot, feature_evo["single"]), axis=1)
        if normalize:
            seq_emd = (seq_emd - min_repr) / (max_repr - min_repr)
        protein_features[id] = seq_emd

    return protein_features


def feature_extraction(
    ID_list, seq_list, conf, device, ion_type="ZN", model_name="ProtTrans"
):
    assert model_name in ["Evoformer", "Composite", "ProtTrans"], "Invalid model name"
    if model_name == "Evoformer":
        assert (
            conf.data.precomputed_feature
        ), "No online Evoformer embedding support yet"
        protein_features = load_evoformer_embedding(
            ID_list,
            conf.data.precomputed_feature,
            normalize=conf.data.normalize,
            ion_type=ion_type,
        )
    elif model_name == "ProtTrans":
        protein_features = prottrans_embedding(
            ID_list,
            seq_list,
            conf,
            device,
            normalize=conf.data.normalize,
            ion_type=ion_type,
        )
    elif model_name == "Composite":
        protein_features = composite_embedding(
            ID_list,
            conf.data.precomputed_feature,
            normalize=conf.data.normalize,
            ion_type=ion_type,
        )

    return protein_features


def data_loader(conf, device, random_seed=0, ion_type="ZN"):
    fasta_path = conf.data.data_path + "/{}_train.fa".format(ion_type)
    ID_list, seq_list, label_list = process_fasta(fasta_path, conf.data.max_seq_len)
    protein_features = feature_extraction(
        ID_list, seq_list, conf, device, ion_type=ion_type, model_name=conf.model.name
    )
    pred_dict = {"ID": ID_list, "Sequence": seq_list, "Label": label_list}
    pred_df = pd.DataFrame(pred_dict)
    pos_weight = calculate_pos_weight(label_list)

    dataset = MetalDataset(pred_df, protein_features, conf.model.feature_dim)
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

    return train_dataloader, val_dataloader, pos_weight


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
