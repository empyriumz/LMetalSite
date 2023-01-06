import datetime
import argparse
import json
import torch
import logging
from tqdm import tqdm
from ml_collections import config_dict
from timeit import default_timer as timer
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
)
from pathlib import Path
from script.utils import (
    logging_related,
    parse_arguments,
)
from data.data_process import prep_dataset, prep_dataloader
from model.model import LMetalSite_Test


def main(conf):
    RANDOM_SEED = int(conf.general.seed)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    device = (
        torch.device("cuda:{}".format(conf.general.gpu_id))
        if torch.cuda.is_available()
        else "cpu"
    )
    logging.info(
        "Test begins at {}".format(datetime.datetime.now().strftime("%m-%d %H:%M"))
    )
    if conf.data.feature == "Evoformer":
        conf.model.feature_dim = 384
    elif conf.data.feature == "ProtTrans":
        conf.model.feature_dim = 1024
    elif conf.data.feature == "Composite":
        conf.model.feature_dim = 1408
    models = []
    for i in range(1):
        model = LMetalSite_Test(
            conf.model.feature_dim,
            conf.model.hidden_dim,
            conf.model.num_encoder_layers,
            conf.model.num_heads,
            conf.model.augment_eps,
            conf.model.dropout,
            conf.model.ligand,
        ).to(device)

        state_dict = torch.load("model/" + "fold{}.ckpt".format(i), device)
        model.load_state_dict(state_dict)
        model.eval()
        model.training = False
        models.append(model)

    metric_auc = BinaryAUROC(thresholds=None)
    metric_auprc = BinaryAveragePrecision(thresholds=None)

    model.training = False  # adding Gaussian noise to embedding
    for ligand in ["MN", "ZN", "MG", "CA"]:
        dataset, _ = prep_dataset(conf, device, ligand=ligand)
        _, val_dataloader = prep_dataloader(
            dataset, conf, random_seed=RANDOM_SEED, ligand=ligand
        )
        with torch.no_grad():
            all_outputs, all_labels = [], []
            for i, batch_data in tqdm(enumerate(val_dataloader)):
                feats, labels, masks = batch_data
                feats = feats.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                model.ligand = ligand
                outputs = [model(feats, masks) for model in models]
                outputs = torch.stack(outputs).mean(0)
                labels = torch.masked_select(labels, masks.bool())
                outputs = torch.sigmoid(torch.masked_select(outputs, masks.bool()))
                all_outputs.append(outputs)
                all_labels.append(labels)
            all_outputs = torch.cat(all_outputs).detach().cpu()
            all_labels = torch.cat(all_labels).detach().cpu()
            test_auc = metric_auc(all_outputs, all_labels)
            test_auprc = metric_auprc(all_outputs, all_labels)
            logging.info(
                "Test auc {:.3f}, auprc: {:.3f}".format(
                    test_auc,
                    test_auprc,
                )
            )
    logging.info(
        "Test is done at {}".format(datetime.datetime.now().strftime("%m-%d %H:%M"))
    )


if __name__ == "__main__":
    start = timer()
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    with open(args.config, "r") as f:
        conf = json.load(f)

    output_path = None
    if not conf["general"]["debug"]:
        output_path = (
            Path("./results/")
            / Path(args.config).stem
            / Path(conf["data"]["feature"] + "_" + conf["model"]["name"])
            / Path(
                str(datetime.datetime.now())[:16].replace(" ", "-").replace(":", "-")
            )
        )
        output_path.mkdir(parents=True, exist_ok=True)
        conf["output_path"] = "./" + str(output_path)
        with open(str(output_path) + "/config.json", "w") as f:
            json.dump(conf, f, indent=4)
    conf = config_dict.ConfigDict(conf)
    """
    logging related part
    """
    logging_related(output_path=output_path, debug=conf.general.debug)
    main(conf)
    end = timer()
    logging.info("Total time used: {:.1f}".format(end - start))
