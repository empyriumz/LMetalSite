import torch
import logging
import string
import re
from Bio import SeqIO
from torch.utils.data import DataLoader
from .utils import calculate_pos_weight
from .datasets import MetalDataset, UniRefDataset, MultiModalDataset, FineTuneDataset
from .embeddings import (
    load_evoformer_embedding,
    prottrans_embedding,
    composite_embedding,
    esm_embedding,
    multimodal_embedding,
)


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


def process_fasta_biopython(fasta_file):
    records = list(SeqIO.parse(fasta_file, "fasta"))
    seq_list, ID_list = [], []
    for rec in records:
        seq_list.append(str(rec.seq))
        ID_list.append(rec.id)
    assert len(ID_list) == len(seq_list), "broken fasta input"
    assert len(seq_list) == len(set(seq_list)), "duplicate entries found"
    del records
    return [ID_list, seq_list]


def feature_extraction(
    ID_list, seq_list, conf, device, ion_type=None, feature_name="ProtTrans"
):
    assert feature_name in [
        "Evoformer",
        "Composite",
        "MultiModal",
        "ProtTrans",
        "ESM",
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
    elif feature_name == "ESM":
        protein_features = esm_embedding(
            ID_list,
            seq_list,
            conf,
            device,
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


def prep_dataset(conf, device, ion_type=None, tokenizer=None):
    assert conf.data.data_type in ["original", "finetune", "multi_modal", "uniref"]
    pos_weight = None
    if conf.data.data_type != "uniref":
        fasta_path = conf.data.data_path + "/{}_train.fa".format(ion_type)

    if conf.data.data_type == "uniref":
        ID_list, seq_list = process_fasta_biopython(conf.data.fasta_path)
    else:
        ID_list, seq_list, label_list = process_fasta(fasta_path, conf.data.max_seq_len)
        pos_weight = calculate_pos_weight(label_list)

    protein_features = feature_extraction(
        ID_list,
        seq_list,
        conf,
        device,
        ion_type=ion_type,
        feature_name=conf.data.feature,
    )
    if conf.data.data_type == "original":
        dataset = MetalDataset(ID_list, seq_list, label_list, protein_features)
    elif conf.data.data_type == "finetune":
        assert tokenizer is not None, "Invalid tokenizer"
        dataset = FineTuneDataset(ID_list, seq_list, label_list, tokenizer)
    elif conf.data.data_type == "multi_modal":
        dataset = MultiModalDataset(ID_list, seq_list, label_list, protein_features)
    elif conf.data.data_type == "uniref":
        dataset = UniRefDataset(ID_list, seq_list, protein_features)

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
