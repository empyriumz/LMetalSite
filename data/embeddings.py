import re
import os
import gc
from tqdm import tqdm
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap
from .utils import load_esm_model

max_repr_evo = np.load("script/Evoformer_repr_max.npy")
min_repr_evo = np.load("script/Evoformer_repr_min.npy")
max_repr_prot = np.load("script/ProtTrans_repr_max.npy")
min_repr_prot = np.load("script/ProtTrans_repr_min.npy")
max_repr_esm = np.load("script/ESM_repr_max.npy")
min_repr_esm = np.load("script/ESM_repr_min.npy")


def esm_embedding(ID_list, seq_list, conf, device, normalize=True, ion_type="ZN"):
    protein_features = {}
    if conf.data.precomputed_feature:
        for id in ID_list:
            tmp = np.load(
                conf.data.precomputed_feature + "/{}_esm/{}.npz".format(ion_type, id)
            )
            seq_emd = tmp["embedding"]
            if normalize:
                seq_emd = (seq_emd - min_repr_esm) / (max_repr_esm - min_repr_esm)
            protein_features[id] = seq_emd

        return protein_features

    if conf.data.save_feature:
        feat_path = conf.output_path + "/{}_esm".format(ion_type)
        os.makedirs(feat_path, exist_ok=True)
    # init the distributed world with world_size 1
    url = "tcp://localhost:23456"
    torch.distributed.init_process_group(
        backend="nccl", init_method=url, world_size=1, rank=0
    )
    # initialize the model with FSDP wrapper
    fsdp_params = dict(
        mixed_precision=True,
        flatten_parameters=True,
        state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
        # state_dict_device=device,
        cpu_offload=True,  # enable cpu offloading
    )

    with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
        model, alphabet = load_esm_model("esm_large")
        # model.to(device)
        batch_converter = alphabet.get_batch_converter()
        model.eval()

        # Wrap each layer in FSDP separately
        for name, child in model.named_children():
            if name == "layers":
                for layer_name, layer in child.named_children():
                    wrapped_layer = wrap(layer)
                    setattr(child, layer_name, wrapped_layer)
        model = wrap(model)

    data = list(zip(ID_list, seq_list))
    batch_size = conf.training.feature_batch_size
    with torch.no_grad():
        for j in tqdm(range(0, len(data), batch_size)):
            partital_data = data[j : j + batch_size]
            batch_labels, _, batch_tokens = batch_converter(partital_data)
            result = model(
                batch_tokens.to(device),
                repr_layers=[len(model.layers)],
                return_contacts=True,
            )
            embedding = (
                result["representations"][len(model.layers)].detach().cpu().numpy()
            )
            mask = result["mask"].detach().cpu().numpy()
            contact = result["contacts"].detach().cpu().numpy()
            for seq_num in range(len(embedding)):
                seq_len = mask[seq_num].sum()
                # get rid of cls and eos token
                seq_emd = embedding[seq_num][1 : (seq_len - 1)]
                # contact prediction already excluded eos and cls
                seq_contact = contact[seq_num][: (seq_len - 2), : (seq_len - 2)]

                if conf.data.save_feature:
                    # np.save(feat_path + "/" + batch_labels[seq_num], seq_emd)
                    # protein_features[batch_labels[seq_num]] = seq_emd
                    np.savez(
                        feat_path + "/" + batch_labels[seq_num],
                        embedding=seq_emd,
                        contact=seq_contact,
                    )
                    protein_features[batch_labels[seq_num]] = [seq_emd, seq_contact]


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


def prottrans_embedding(ID_list, seq_list, conf, device, normalize=True, ion_type=None):
    protein_features = {}
    if conf.data.precomputed_feature:

        for id in ID_list:
            if ion_type is not None:
                file_name = "/{}_ProtTrans/{}.npy".format(ion_type, id)
            else:
                file_name = "/{}.npy".format(id)
            seq_emd = np.load(conf.data.precomputed_feature + file_name)
            if normalize:
                seq_emd = (seq_emd - min_repr_prot) / (max_repr_prot - min_repr_prot)
            protein_features[id] = seq_emd

        return protein_features

    if conf.data.save_feature:
        feat_path = conf.output_path + "/ProtTrans"
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

        # Load sequences and map rarely occurred amino acids (U,Z,O,B) to (X)
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


def multimodal_embedding_contact_map(
    ID_list, precomputed_feature_path, normalize=True, ion_type="ZN"
):
    protein_features = {}
    for id in ID_list:
        tmp = np.load(precomputed_feature_path + "/{}_esm/{}.npz".format(ion_type, id))
        seq_emd = tmp["embedding"]
        contact_map = tmp["contact"]
        if normalize:
            seq_emd = (seq_emd - min_repr_esm) / (max_repr_esm - min_repr_esm)
        pad_shape = 1000 - contact_map.shape[0]
        contact_map = np.pad(contact_map, ((0, pad_shape), (0, pad_shape)), "constant")
        # seq_emd = np.pad(seq_emd, ((0, pad_shape), (0, 0)), 'constant')
        protein_features[id] = [seq_emd, contact_map]

    return protein_features
