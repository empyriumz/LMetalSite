import os, datetime, argparse, re
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer
from torch.utils.data import DataLoader
import gc
from model.model import LMetalSite_Test
from data.data_process import MetalDatasetTest
from data.utils import process_fasta

############ Set to your own path! ############
# ProtTrans_path = "/home/yuanqm/protein_binding_sites/tools/Prot-T5-XL-U50"

model_path = "./model/"

Max_repr = np.load("script/ProtTrans_repr_max.npy")
Min_repr = np.load("script/ProtTrans_repr_min.npy")

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


def feature_extraction(ID_list, seq_list, outpath, feat_bs, save_feat, device):
    protein_features = {}
    if save_feat:
        feat_path = outpath + "ProtTrans_repr"
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

    # Extract feature of one batch each time
    for i in tqdm(range(0, len(ID_list), feat_bs)):
        if i + feat_bs <= len(ID_list):
            batch_ID_list = ID_list[i : i + feat_bs]
            batch_seq_list = seq_list[i : i + feat_bs]
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
            if save_feat:
                np.save(feat_path + "/" + batch_ID_list[seq_num], seq_emd)
            # Normalization
            seq_emd = (seq_emd - Min_repr) / (Max_repr - Min_repr)
            protein_features[batch_ID_list[seq_num]] = seq_emd

    return protein_features


def predict(
    run_id, ID_list, seq_list, protein_features, config, outpath, pred_bs, device
):
    pred_dict = {ID_col: ID_list, sequence_col: seq_list}
    pred_df = pd.DataFrame(pred_dict)

    for metal in metal_list:
        pred_df[metal + "_prob"] = 0.0
        pred_df[metal + "_pred"] = 0.0

    test_dataset = MetalDatasetTest(pred_df, protein_features)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=pred_bs,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    # Load LMetalSite models
    models = []
    for i in range(5):
        model = LMetalSite_Test(
            config["feature_dim"],
            config["hidden_dim"],
            config["num_encoder_layers"],
            config["num_heads"],
            config["augment_eps"],
            config["dropout"],
        ).to(device)

        state_dict = torch.load(model_path + "fold{}.ckpt".format(i), device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    # Make Predictions
    prob_col = [[] for _ in range(len(metal_list))]
    pred_col = [[] for _ in range(len(metal_list))]
    with torch.no_grad():
        for batch_data in tqdm(test_dataloader):
            protein_feats, protein_masks, maxlen = batch_data
            protein_feats = protein_feats.to(device)
            protein_masks = protein_masks.to(device)

            outputs = [
                model(protein_feats, protein_masks).sigmoid() for model in models
            ]
            outputs = torch.stack(outputs).mean(
                0
            )  # average the predictions from 5 models
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

    pred_df.to_csv(outpath + run_id + "_predictions.csv", index=False)


def main(run_id, seq_info, outpath, feat_bs, pred_bs, save_feat, gpu):
    ID_list, seq_list = seq_info
    # device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    device = torch.device("cuda:{}".format(gpu)) if torch.cuda.is_available() else "cpu"

    print(
        "\n######## Feature extraction begins at {}. ########\n".format(
            datetime.datetime.now().strftime("%m-%d %H:%M")
        )
    )

    protein_features = feature_extraction(
        ID_list, seq_list, outpath, feat_bs, save_feat, device
    )

    print(
        "\n######## Feature extraction is done at {}. ########\n".format(
            datetime.datetime.now().strftime("%m-%d %H:%M")
        )
    )

    print(
        "\n######## Prediction begins at {}. ########\n".format(
            datetime.datetime.now().strftime("%m-%d %H:%M")
        )
    )

    predict(
        run_id, ID_list, seq_list, protein_features, NN_config, outpath, pred_bs, device
    )

    print(
        "\n######## Prediction is done at {}. ########\n".format(
            datetime.datetime.now().strftime("%m-%d %H:%M")
        )
    )
    print("Results are saved in {}".format(outpath + run_id + "_predictions.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=str, help="Input fasta file")
    parser.add_argument(
        "--outpath",
        type=str,
        help="Output path to save intermediate features and final predictions",
    )
    parser.add_argument(
        "--feat_bs",
        type=int,
        default=10,
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
    outpath = args.outpath.rstrip("/") + "/"

    run_id = args.fasta.split("/")[-1].split(".")[0]
    seq_info = process_fasta(args.fasta, MAX_INPUT_SEQ)

    if seq_info == -1:
        print("The format of your input fasta file is incorrect! Please check!")
    elif seq_info == 1:
        print(
            "Too much sequences! Up to {} sequences are supported each time!".format(
                MAX_INPUT_SEQ
            )
        )
    else:
        main(
            run_id,
            seq_info,
            outpath,
            args.feat_bs,
            args.pred_bs,
            args.save_feat,
            args.gpu,
        )
