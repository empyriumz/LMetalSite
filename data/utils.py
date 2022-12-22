import numpy as np
import torch
import re
import string
import os
import gc
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

max_repr_evo = np.load("script/Evoformer_repr_max.npy")
min_repr_evo = np.load("script/Evoformer_repr_min.npy")
max_repr_prot = np.load("script/ProtTrans_repr_max.npy")
min_repr_prot = np.load("script/ProtTrans_repr_min.npy")


def composite_embedding(
    ID_list, precomputed_feature_path, normalize=True, ion_type="ZN"
):
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


def load_evoformer_embedding(
    ID_list, precomputed_feature_path, normalize=True, ion_type="ZN"
):
    protein_features = {}
    for id in ID_list:
        feature = np.load(
            precomputed_feature_path + "/{}_Evoformer/{}.npz".format(ion_type, id)
        )
        # seq_emd = np.concatenate((feature["single"], feature["pair"]), axis=1)
        seq_emd = feature["single"]
        # seq_emd = feature["pair"]
        if normalize:
            seq_emd = (seq_emd - min_repr_evo) / (max_repr_evo - min_repr_evo)
        protein_features[id] = seq_emd

    return protein_features


def prottrans_embedding(ID_list, seq_list, conf, device, normalize=True, ion_type="ZN"):
    protein_features = {}
    if conf.data.precomputed_feature:
        for id in ID_list:
            seq_emd = np.load(
                conf.data.precomputed_feature
                + "/{}_ProtTrans/{}.npy".format(ion_type, id)
            )
            if normalize:
                seq_emd = (seq_emd - min_repr_prot) / (max_repr_prot - min_repr_prot)
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
            seq_emd = (seq_emd - min_repr_prot) / (max_repr_prot - min_repr_prot)
            protein_features[batch_ID_list[seq_num]] = seq_emd

    return protein_features


def multimodal_embedding(
    ID_list, precomputed_feature_path, normalize=True, ion_type="ZN"
):
    protein_features = {}
    for id in ID_list:
        feature_evo = np.load(
            precomputed_feature_path + "/{}_Evoformer/{}.npz".format(ion_type, id)
        )
        feature_prot = np.load(
            precomputed_feature_path + "/{}_ProtTrans/{}.npy".format(ion_type, id)
        )
        if normalize:
            feature_prot = (feature_prot - min_repr_prot) / (
                max_repr_prot - min_repr_prot
            )
            feature_evo = (feature_evo["single"] - min_repr_evo) / (
                max_repr_evo - min_repr_evo
            )

        protein_features[id] = [feature_prot, feature_evo]

    return protein_features


def feature_extraction(
    ID_list, seq_list, conf, device, ion_type="ZN", feature_name="ProtTrans"
):
    assert feature_name in [
        "Evoformer",
        "Composite",
        "MultiModal",
        "ProtTrans",
    ], "Invalid feature name"
    if feature_name == "Evoformer":
        assert (
            conf.data.precomputed_feature
        ), "No online Evoformer embedding support yet"
        protein_features = load_evoformer_embedding(
            ID_list,
            conf.data.precomputed_feature,
            normalize=conf.data.normalize,
            ion_type=ion_type,
        )
    elif feature_name == "ProtTrans":
        protein_features = prottrans_embedding(
            ID_list,
            seq_list,
            conf,
            device,
            normalize=conf.data.normalize,
            ion_type=ion_type,
        )
    elif feature_name == "Composite":
        protein_features = composite_embedding(
            ID_list,
            conf.data.precomputed_feature,
            normalize=conf.data.normalize,
            ion_type=ion_type,
        )
    elif feature_name == "MultiModal":
        protein_features = multimodal_embedding(
            ID_list,
            conf.data.precomputed_feature,
            normalize=conf.data.normalize,
            ion_type=ion_type,
        )

    return protein_features


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
