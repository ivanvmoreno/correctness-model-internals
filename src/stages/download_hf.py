import argparse
import os
import sys

import huggingface_hub as hf

from src.utils.config import load_config
from src.utils.logging import get_logger

"""
Reads 'params.yaml' to determine which datasets and models to download from HuggingFace.
Secrets (like HuggingFace tokens) are handled via environment variables.

Usage:
  python scripts/download_hf.py

Expects the following environment variable(s):
  HF_AUTH_TOKEN (optional): The Hugging Face access token if private repositories or more quota are needed.
"""


def download_hf(config_path: str, model: str = None) -> None:
    config = load_config(config_path)
    logger = get_logger("DOWNLOAD_HF", config.base.log_level)

    logger.info("Logging into Hugging Face...")
    hf_token = os.getenv("HF_AUTH_TOKEN")
    if not hf_token:
        logger.error(
            "No Hugging Face token provided. Please set the 'HF_AUTH_TOKEN' environment variable."
        )
        sys.exit(1)
    hf.login(hf_token)

    logger.info("Downloading datasets from Hugging Face...")
    for dataset_name, dataset_conf in config.datasets.items():
        if not hasattr(dataset_conf, "hf_repo_id"):
            logger.warning(
                f"Missing 'hf_repo_id' key for dataset '{dataset_name}'. Considering it as a local dataset."
            )
            continue

        logger.info(
            f"Downloading dataset '{dataset_name}' from Hugging Face repo '{dataset_conf.hf_repo_id}'"
        )
        hf.snapshot_download(
            repo_id=dataset_conf.hf_repo_id,
            token=hf_token,
            repo_type="dataset",
            local_dir=f"{config.base.datasets_dir}/{config.format_datasets.raw_dir_path}/{dataset_name}",
        )

    logger.info("Downloading model(s) from Hugging Face...")
    models = [model] if model else config.generate_answers.models
    for model_name in models:
        model_conf = config.models[model_name]
        if not hasattr(model_conf, "hf_repo_id"):
            logger.warning(
                f"Missing 'hf_repo_id' key for model '{model_name}'. Skipping download."
            )
            continue

        logger.info(
            f"Downloading model '{model_name}' from Hugging Face repo '{model_conf.hf_repo_id}'"
        )
        hf.snapshot_download(
            repo_id=model_conf.hf_repo_id,
            token=hf_token,
            repo_type="model",
            local_dir=f"{config.base.models_dir}/{model_name}",
        )


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument("--model", dest="model", default=None)
    args = args_parser.parse_args()

    download_hf(args.config, args.model)
