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
    add_alphafold_args,
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
    if conf.model.name == "Evoformer":
        conf.model.feature_dim = 384
    elif conf.model.name == "ProtTrans":
        conf.model.feature_dim = 1024
    elif conf.model.name == "Composite":
        conf.model.feature_dim = 1408
    models = []
    for i in range(5):
        model = LMetalSite_Test(
            conf.model.feature_dim,
            conf.model.hidden_dim,
            conf.model.num_encoder_layers,
            conf.model.num_heads,
            conf.model.augment_eps,
            conf.model.dropout,
            conf.model.ion_type,
        ).to(device)

        state_dict = torch.load("model/" + "fold{}.ckpt".format(i), device)
        model.load_state_dict(state_dict)
        model.eval()
        model.training = False
        models.append(model)

    metric_auc = BinaryAUROC(thresholds=None)
    metric_auprc = BinaryAveragePrecision(thresholds=None)

    model.training = False  # adding Gaussian noise to embedding
    for ion in ["MN", "ZN", "MG", "CA"]:
        dataset, _ = prep_dataset(conf, device, ion_type=ion)
        _, val_dataloader = prep_dataloader(
            dataset, conf, random_seed=RANDOM_SEED, ion_type=ion
        )
        with torch.no_grad():
            all_outputs, all_labels = [], []
            for i, batch_data in tqdm(enumerate(val_dataloader)):
                feats, labels, masks, _ = batch_data
                feats = feats.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                model.ion_type = ion
                outputs = [model(feats, masks) for model in models]
                outputs = torch.stack(outputs).mean(0)
                labels = torch.masked_select(labels, masks.bool())
                outputs = torch.sigmoid(torch.masked_select(outputs, masks.bool()))
                all_outputs.append(outputs)
                all_labels.append(labels)
            all_outputs = torch.cat(all_outputs)
            all_labels = torch.cat(all_labels)
            test_auc = metric_auc(all_outputs, all_labels).detach().cpu().numpy()
            test_auprc = metric_auprc(all_outputs, all_labels).detach().cpu().numpy()
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
    add_alphafold_args(parser)
    output_path = (
        Path("./results/")
        / Path(args.config).stem
        / Path(str(datetime.datetime.now())[:16].replace(" ", "-").replace(":", "-"))
    )
    output_path.mkdir(parents=True, exist_ok=True)

    """
    Read configuration and dump the configuration to output dir
    """
    with open(args.config, "r") as f:
        conf = json.load(f)
    conf["output_path"] = "./" + str(output_path)
    with open(str(output_path) + "/config.json", "w") as f:
        json.dump(conf, f, indent=4)

    conf = config_dict.ConfigDict(conf)
    """
    logging related part
    """
    logging_related(output_path)
    main(conf)
    end = timer()
    logging.info("Total time used: {:.1f}".format(end - start))
