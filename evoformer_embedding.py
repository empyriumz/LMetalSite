# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import math
import numpy as np
import os

from openfold.utils.script_utils import run_model
from openfold.utils.import_weights import (
    import_evoformer_weights_,
)
from openfold.model.evoformer_inference import Evoformer

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

import random
import time
import torch
from timeit import default_timer as timer

start = timer()
torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if torch_major_version > 1 or (torch_major_version == 1 and torch_minor_version >= 12):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

torch.set_grad_enabled(False)

from openfold.config import model_config
from openfold.data import feature_pipeline, data_pipeline
from data.data_process import process_train_fasta

from openfold.utils.tensor_utils import (
    tensor_tree_map,
)
from openfold.utils.trace_utils import (
    pad_feature_dict_seq,
    trace_model_,
)
from script.utils import add_alphafold_args


TRACING_INTERVAL = 50


def precompute_alignments(tags, seqs, alignment_dir, args):
    for tag, seq in zip(tags, seqs):
        tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(alignment_dir, tag)
        if args.use_precomputed_alignments is None and not os.path.isdir(
            local_alignment_dir
        ):
            logger.info(f"Generating alignments for {tag}...")

            os.makedirs(local_alignment_dir)

            alignment_runner = data_pipeline.AlignmentRunner(
                jackhmmer_binary_path=args.jackhmmer_binary_path,
                hhblits_binary_path=args.hhblits_binary_path,
                hhsearch_binary_path=args.hhsearch_binary_path,
                uniref90_database_path=args.uniref90_database_path,
                mgnify_database_path=args.mgnify_database_path,
                bfd_database_path=args.bfd_database_path,
                uniclust30_database_path=args.uniclust30_database_path,
                pdb70_database_path=args.pdb70_database_path,
                no_cpus=args.cpus,
            )
            alignment_runner.run(tmp_fasta_path, local_alignment_dir)
        else:
            logger.info(f"Using precomputed alignments for {tag} at {alignment_dir}...")

        # Remove temporary FASTA file
        os.remove(tmp_fasta_path)


def round_up_seqlen(seqlen):
    return int(math.ceil(seqlen / TRACING_INTERVAL)) * TRACING_INTERVAL


def run_model(model, batch, tag):
    with torch.no_grad():
        logger.info(f"Running inference for {tag}...")
        t = time.perf_counter()
        out = model(batch)
        inference_time = time.perf_counter() - t
        logger.info("Inference time: {:.1f}".format(inference_time))

    return out


def generate_feature_dict(
    tags,
    seqs,
    alignment_dir,
    data_processor,
    args,
):
    tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
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


def get_model_basename(model_path):
    return os.path.splitext(os.path.basename(os.path.normpath(model_path)))[0]


def make_output_directory(output_dir, model_name, multiple_model_mode):
    if multiple_model_mode:
        prediction_dir = os.path.join(output_dir, "predictions", model_name)
    else:
        prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    return prediction_dir


def main(args):
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model_name
    config = model_config(model_name)
    model = Evoformer(config)
    model = model.eval()
    npz_path = os.path.join(args.jax_param_path, "params_" + model_name + ".npz")
    import_evoformer_weights_(model, npz_path, version=model_name)
    model = model.to(args.model_device)
    if args.trace_model:
        if not config.data.predict.fixed_size:
            raise ValueError(
                "Tracing requires that fixed_size mode be enabled in the config"
            )
    template_featurizer = None
    data_processor = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )

    output_dir_base = args.output_dir
    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2**32)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)

    feature_processor = feature_pipeline.FeaturePipeline(config.data)
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)
    if args.use_precomputed_alignments is None:
        alignment_dir = os.path.join(output_dir_base, "alignments")
    else:
        alignment_dir = args.use_precomputed_alignments

    ID_list, seq_list, _ = process_train_fasta(args.fasta_dir, args.max_seq_len)
    sorted_targets = list(zip(*(ID_list, seq_list)))
    feature_dicts = {}
    cur_tracing_interval = 0
    for tag, seq in sorted_targets:
        feature_dict = feature_dicts.get(tag, None)
        if feature_dict is None:
            feature_dict = generate_feature_dict(
                [tag],
                [seq],
                alignment_dir,
                data_processor,
                args,
            )
            if args.trace_model:
                n = feature_dict["aatype"].shape[-2]
                rounded_seqlen = round_up_seqlen(n)
                feature_dict = pad_feature_dict_seq(
                    feature_dict,
                    rounded_seqlen,
                )
            feature_dicts[tag] = feature_dict

        processed_feature_dict = feature_processor.process_features(
            feature_dict,
            mode="predict",
        )

        processed_feature_dict = {
            k: torch.as_tensor(v, device=args.model_device)
            for k, v in processed_feature_dict.items()
        }

        if args.trace_model:
            if rounded_seqlen > cur_tracing_interval:
                logger.info(f"Tracing model at {rounded_seqlen} residues...")
                t = time.perf_counter()
                trace_model_(model, processed_feature_dict)
                tracing_time = time.perf_counter() - t
                logger.info(f"Tracing time: {tracing_time}")
                cur_tracing_interval = rounded_seqlen

        # Toss out the recycling dimensions
        processed_feature_dict = tensor_tree_map(
            lambda x: x[..., -1], processed_feature_dict
        )
        out = run_model(model, processed_feature_dict, tag)
        out = tensor_tree_map(lambda x: np.array(x.cpu()), out)
        if args.save_outputs:
            np.savez_compressed(
                args.output_dir + "/" + tag,
                single=out["single"],
                pair=out["pair"][..., 0, :, :],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta_dir",
        type=str,
        help="Path to directory containing FASTA files, one sequence per file",
    )
    parser.add_argument(
        "template_mmcif_dir",
        type=str,
    )
    parser.add_argument(
        "--use_precomputed_alignments",
        type=str,
        default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored.""",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="model_1",
        help="""Name of a model config preset defined in openfold/config.py""",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getcwd(),
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--model_device",
        type=str,
        default="cpu",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")""",
    )
    parser.add_argument(
        "--config_preset",
        type=str,
        default="model_1",
        help="""Name of a model config preset defined in openfold/config.py""",
    )
    parser.add_argument(
        "--jax_param_path",
        type=str,
        default="/hpcgpfs01/scratch/xdai/openfold/params",
        help="""Path to JAX model parameters. If None, and openfold_checkpoint_path
             is also None, parameters are selected automatically according to 
             the model name from openfold/resources/params""",
    )
    parser.add_argument(
        "--openfold_checkpoint_path",
        type=str,
        default=None,
        help="""Path to OpenFold checkpoint. Can be either a DeepSpeed 
             checkpoint directory or a .pt file""",
    )
    parser.add_argument(
        "--save_outputs",
        default=0,
        type=int,
        help="Whether to save all model outputs, including embeddings, etc.",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=8,
        help="""Number of CPUs with which to run alignment tools""",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=5000,
        help="""Maximum sequence length""",
    )
    parser.add_argument(
        "--preset", type=str, default="full_dbs", choices=("reduced_dbs", "full_dbs")
    )
    parser.add_argument(
        "--output_postfix",
        type=str,
        default=None,
        help="""Postfix for output prediction filenames""",
    )
    parser.add_argument("--data_random_seed", type=str, default=None)
    parser.add_argument(
        "--multimer_ri_gap",
        type=int,
        default=200,
        help="""Residue index offset between multiple sequences, if provided""",
    )
    parser.add_argument(
        "--trace_model",
        default=0,
        type=int,
        help="""Whether to convert parts of each model to TorchScript.
                Significantly improves runtime at the cost of lengthy
                'compilation.' Useful for large batch jobs.""",
    )
    add_alphafold_args(parser)
    args = parser.parse_args()

    if args.model_device == "cpu" and torch.cuda.is_available():
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )
    main(args)
    end = timer()
    logger.info("total time used {:.2f}".format(end - start))
