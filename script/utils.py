import argparse
from datetime import date
import logging
import sys


def parse_arguments(parser):
    parser.add_argument(
        "--config", type=str, default=None, help="Name of configuration file"
    )
    parser.add_argument("--fasta", type=str, help="Input fasta file")
    parser.add_argument(
        "--outpath",
        type=str,
        help="Output path to save intermediate features and final predictions",
    )
    parser.add_argument(
        "--feat_bs",
        type=int,
        default=8,
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
    return args


def add_alphafold_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--uniref90_database_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--mgnify_database_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pdb70_database_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--uniclust30_database_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--bfd_database_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--jackhmmer_binary_path", type=str, default="/usr/bin/jackhmmer"
    )
    parser.add_argument(
        "--hhblits_binary_path",
        type=str,
        default="/opt/conda/envs/openfold/bin/hhblits",
    )
    parser.add_argument(
        "--hhsearch_binary_path",
        type=str,
        default="/opt/conda/envs/openfold/bin/hhsearch",
    )
    parser.add_argument(
        "--kalign_binary_path", type=str, default="/opt/conda/envs/openfold/bin/kalign"
    )
    parser.add_argument(
        "--max_template_date",
        type=str,
        default=date.today().strftime("%Y-%m-%d"),
    )
    parser.add_argument("--obsolete_pdbs_path", type=str, default=None)
    parser.add_argument("--release_dates_path", type=str, default=None)


def logging_related(output_path=None, debug=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if not debug:
        assert output_path is not None, "need valid log output path"
        log_filename = str(output_path) + "/training.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logging.info("Output path: {}".format(output_path))
