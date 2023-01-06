import datetime
import argparse
import json
import torch
import logging
from tqdm import tqdm
from ml_collections import config_dict
from timeit import default_timer as timer
from pathlib import Path
from script.utils import (
    logging_related,
    parse_arguments,
)
from data.data_process import prep_dataset, prep_dataloader
from model.model import LMetalSiteEncoder

LOG_INTERVAL = 50


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
        "Training begins at {}".format(datetime.datetime.now().strftime("%m-%d %H:%M"))
    )
    if conf.data.feature == "Evoformer":
        conf.model.feature_dim = 384
    elif conf.data.feature == "ProtTrans":
        conf.model.feature_dim = 1024
    elif conf.data.feature == "Composite":
        conf.model.feature_dim = 1408
    elif conf.data.feature == "ESM":
        conf.model.feature_dim = 2560
    else:
        raise ValueError("No feature available")

    model = LMetalSiteEncoder(conf.model).to(device)

    if conf.training.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            betas=(0.9, 0.99),
            lr=conf.training.learning_rate,
            weight_decay=conf.training.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=conf.training.learning_rate,
            weight_decay=conf.training.weight_decay,
        )
    best_loss = 1000
    model.training = True  # adding Gaussian noise to embedding
    for ligand in ["ZN", "MN", "MG", "CA"]:
        dataset, _ = prep_dataset(conf, device, ligand=ligand)
        train_dataloader, val_dataloader = prep_dataloader(
            dataset, conf, random_seed=RANDOM_SEED, ligand=ligand
        )
        loss_func = torch.nn.MSELoss()
        model.ligand = ligand
        for epoch in range(conf.training.epochs):
            train_loss = 0.0
            for i, batch_data in tqdm(enumerate(train_dataloader)):
                feats, _, _, _ = batch_data
                optimizer.zero_grad(set_to_none=True)
                feats = feats.to(device)
                outputs = model(feats)
                loss_ = loss_func(outputs, feats)
                loss_.backward()
                optimizer.step()
                running_loss = loss_.detach().cpu().numpy()
                train_loss += running_loss
                if i % LOG_INTERVAL == 0 and i > 0:
                    logging.info("Running train loss: {:.4f}".format(running_loss))
            logging.info(
                "Epoch {} train loss {:.4f}".format(
                    epoch + 1,
                    train_loss / (i + 1),
                )
            )
            model.eval()
            with torch.no_grad():
                model.training = False
                val_loss = 0.0
                for i, batch_data in tqdm(enumerate(val_dataloader)):
                    feats, _, _, _ = batch_data
                    feats = feats.to(device)
                    outputs = model(feats)
                    loss_ = loss_func(outputs, feats)
                    running_loss = loss_.detach().cpu().numpy()
                    val_loss += running_loss

            val_loss = val_loss / (i + 1)
            logging.info(
                "Epoch {} val loss {:.4f}".format(
                    epoch + 1,
                    val_loss,
                )
            )
            if val_loss < best_loss and val_loss < 0.0025 and not conf.general.debug:
                best_loss = val_loss
                state = {
                    "encoder_state": model.input_block.state_dict(),
                }
                file_name = (
                    conf.output_path
                    + "/"
                    + "autoencoder_{}_epoch_{}".format(conf.data.feature, epoch + 1)
                    + "_loss_{:.3f}".format(val_loss)
                )
                torch.save(state, file_name + ".pt")
                logging.info("\n------------ Save the best model ------------\n")

    logging.info(
        "Training is done at {}".format(datetime.datetime.now().strftime("%m-%d %H:%M"))
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
