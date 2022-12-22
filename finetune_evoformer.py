import argparse
import logging
import numpy as np
import os
import random
import time
import torch
import datetime
import json
from pathlib import Path
from ml_collections import config_dict
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
)
from openfold.utils.import_weights import (
    import_evoformer_weights_,
)
from openfold.model.evoformer_inference import Evoformer
from model.finetune_model import MetalIonSiteEvoformer

from timeit import default_timer as timer
from script.utils import (
    logging_related,
    parse_arguments,
)

torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if torch_major_version > 1 or (torch_major_version == 1 and torch_minor_version >= 12):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")


from openfold.config import model_config
from openfold.data import feature_pipeline, data_pipeline
from data.utils import process_fasta

from openfold.utils.tensor_utils import (
    tensor_tree_map,
)


def generate_feature_dict(
    tags,
    seqs,
    alignment_dir,
    data_processor,
    output_dir,
):
    tmp_fasta_path = os.path.join(output_dir, f"tmp_{os.getpid()}.fasta")
    if len(seqs) == 1:
        tag = tags[0]
        seq = seqs[0]
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(alignment_dir, tag)
        feature_dict = data_processor.process_fasta(
            fasta_path=tmp_fasta_path, alignment_dir=local_alignment_dir
        )
    else:
        with open(tmp_fasta_path, "w") as fp:
            fp.write("\n".join([f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)]))
        feature_dict = data_processor.process_multiseq_fasta(
            fasta_path=tmp_fasta_path,
            super_alignment_dir=alignment_dir,
        )

    # Remove temporary FASTA file
    os.remove(tmp_fasta_path)

    return feature_dict


def main(conf):
    device = (
        torch.device("cuda:{}".format(conf.general.gpu_id))
        if torch.cuda.is_available()
        else "cpu"
    )
    model_name = conf.model.alphafold_model
    evoformer_config = model_config(model_name, train=True, low_prec=True)
    backbone_model = Evoformer(evoformer_config)
    npz_path = os.path.join(conf.model.jax_param_path, "params_" + model_name + ".npz")
    import_evoformer_weights_(backbone_model, npz_path, version=model_name)
    model = MetalIonSiteEvoformer(backbone_model, conf).to(device)
    for para in model.parameters():
        para.requires_grad_(True)
    template_featurizer = None
    data_processor = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )

    RANDOM_SEED = int(conf.general.seed)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    feature_processor = feature_pipeline.FeaturePipeline(evoformer_config.data)
    alignment_dir = conf.data.precomputed_alignments_path
    if conf.training.fix_backbone_weight:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=conf.training.learning_rate,
            weight_decay=conf.training.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.classifier.parameters()},
                {
                    "params": model.backbone.parameters(),
                    "lr": conf.training.backbone_learning_rate,
                },
            ],
            lr=conf.training.learning_rate,
            weight_decay=conf.training.weight_decay,
        )

    metric_auc = BinaryAUROC(thresholds=None)
    metric_auprc = BinaryAveragePrecision(thresholds=None)
    for ion in ["CA"]:
        fasta_path = conf.data.fasta_path + "/{}_train.fa".format(ion)
        ID_list, seq_list, label_list = process_fasta(fasta_path)
        sorted_targets = list(zip(*(ID_list, seq_list)))
        feature_dicts = {}
        loss_func = torch.nn.BCEWithLogitsLoss()
        for tag, seq in sorted_targets:
            feature_dict = feature_dicts.get(tag, None)
            if feature_dict is None:
                feature_dict = generate_feature_dict(
                    [tag],
                    [seq],
                    alignment_dir,
                    data_processor,
                    conf.output_path,
                )
                feature_dicts[tag] = feature_dict

            processed_feature_dict = feature_processor.process_features(
                feature_dict,
                mode="predict",
            )

            processed_feature_dict = {
                k: torch.as_tensor(v, device=device)
                for k, v in processed_feature_dict.items()
            }
            # Toss out the recycling dimensions
            processed_feature_dict = tensor_tree_map(
                lambda x: x[..., -1], processed_feature_dict
            )
            out = model(processed_feature_dict)
            out = tensor_tree_map(lambda x: np.array(x.cpu()), out)


if __name__ == "__main__":
    start = timer()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Name of configuration file"
    )
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
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
    logging_related(output_path)
    main(conf)
    end = timer()
    logging.info("total time used {:.2f}".format(end - start))
