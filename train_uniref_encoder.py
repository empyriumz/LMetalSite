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
from model.model import LMetalSiteEncoder, LMetalSiteTransformerEncoder
from torch.utils.tensorboard import SummaryWriter


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

    if conf.model.name == "base":
        model = LMetalSiteEncoder(conf.model).to(device)
    elif conf.model.name == "transformer":
        model = LMetalSiteTransformerEncoder(conf.model).to(device)
    if conf.training.pretrained_encoder:
        checkpoint = torch.load(conf.training.pretrained_encoder)
        logging.info("load weights from {}".format(conf.training.pretrained_encoder))
        model.params.encoder.load_state_dict(checkpoint["encoder_state"])
        model.params.decoder.load_state_dict(checkpoint["decoder_state"])
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
    # loss_func = torch.nn.MSELoss()
    loss_func = torch.nn.SmoothL1Loss(beta=0.2)
    for j in range(3):
        conf.data.fasta_path = (
            "datasets/uniref_sample_plus_metal/sample_plus_metal.fasta.split/sample_plus_metal.part_00{}.fasta".format(j+1)
        )
        dataset, _ = prep_dataset(conf, device)
        train_dataloader, val_dataloader = prep_dataloader(
            dataset, conf, random_seed=RANDOM_SEED
        )
        for epoch in range(1, conf.training.epochs + 1):
            train_loss = 0.0
            for i, batch_data in tqdm(enumerate(train_dataloader)):
                (
                    feats,
                    mask,
                ) = batch_data
                optimizer.zero_grad(set_to_none=True)
                feats = feats.to(device)
                mask = mask.to(device)
                outputs = model(feats, mask)
                outputs = torch.masked_select(outputs, mask.unsqueeze(-1).bool())
                feats = torch.masked_select(feats, mask.unsqueeze(-1).bool())
                loss_ = loss_func(outputs, feats)
                loss_.backward()
                optimizer.step()
                running_loss = loss_.detach().cpu().numpy()
                train_loss += running_loss
            train_loss = train_loss / (i + 1)
            sub_epoch = j * conf.training.epochs + epoch
            logging.info(
                "Epoch {} train loss {:.4f}".format(
                    sub_epoch,
                    train_loss,
                )
            )
            writer.add_scalar("train_loss", train_loss, sub_epoch)
            model.eval()
            with torch.no_grad():
                model.training = False
                val_loss = 0.0
                for i, batch_data in tqdm(enumerate(val_dataloader)):
                    (
                        feats,
                        mask,
                    ) = batch_data
                    feats = feats.to(device)
                    mask = mask.to(device)
                    outputs = model(feats, mask)
                    outputs = torch.masked_select(outputs, mask.unsqueeze(-1).bool())
                    feats = torch.masked_select(feats, mask.unsqueeze(-1).bool())
                    loss_ = loss_func(outputs, feats)
                    running_loss = loss_.detach().cpu().numpy()
                    val_loss += running_loss

            val_loss = val_loss / (i + 1)
            logging.info(
                "Epoch {} val loss {:.4f}".format(
                    sub_epoch,
                    val_loss,
                )
            )
            writer.add_scalar("val_loss", val_loss, sub_epoch)
            if val_loss < best_loss and val_loss < 0.052 and not conf.general.debug:
                best_loss = val_loss
                state = {
                    "encoder_state": model.params.encoder.state_dict(),
                    "decoder_state": model.params.decoder.state_dict(),
                }
                file_name = (
                    conf.output_path
                    + "/"
                    + "autoencoder_{}_epoch_{}".format(conf.data.feature, sub_epoch)
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
    writer = SummaryWriter(log_dir=output_path)
    main(conf)
    writer.flush()
    end = timer()
    logging.info("Total time used: {:.1f}".format(end - start))
