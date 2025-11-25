from typing import Union, List, Optional

import argparse
import os
import gzip
import shutil

from src.data import format_generic, format_gsm8k, format_mmlu, format_notable
from src.utils.config import load_config
from src.utils.logging import get_logger


def format_dataset(
    config_path: str,
    model: Optional[Union[str, List[str]]],
) -> None:
    """Format datasets for question-answering tasks

    Args:
        config_path (str): Path to configuration file
        model (Optional[Union[str, List[str]]]): Model ID(s) to format datasets for.
            If None, all models in params.generate_answers.models are used.
    """
    config = load_config(config_path)
    logger = get_logger("DATA_PREPROCESS", config.base.log_level)

    logger.info(f"Loading datasets from local path {config.base.datasets_dir}")

    if model is None:
        logger.info(
            "No model argument provided. Using all models from params.generate_answers.models"
        )
        model = config.generate_answers.models

    if isinstance(model, str):
        model = [model]

    for model_id in model:
        model_config = config.models[model_id]
        logger.info(f"Formatting datasets for model {model_id}")
        for dataset_name, dataset_conf in config.datasets.items():
            if "compressed" in dataset_conf:
                if dataset_conf.compressed.format == "gzip":
                    raw_path = (
                        f"{config.base.datasets_dir}/"
                        f"{config.format_datasets.raw_dir_path}/"
                        f"{dataset_name}/{dataset_conf.compressed.target}"
                    )
                    if not os.path.isfile(raw_path):
                        logger.info(
                            f"Compressed file {raw_path} does not exist. Skipping."
                        )
                    else:
                        # Only supports 1 subset for compressed datasets
                        subset = (
                            dataset_conf.subsets[0]
                            if dataset_conf.subsets
                            else "main"
                        )
                        formatted_path = (
                            f"{config.base.datasets_dir}/"
                            f"{config.format_datasets.raw_dir_path}/"
                            f"{dataset_name}/{subset}/"
                            f"{dataset_conf.compressed.target[:-3]}"
                        )
                        if not os.path.exists(os.path.dirname(formatted_path)):
                            os.makedirs(
                                os.path.dirname(formatted_path), exist_ok=True
                            )
                        logger.info(
                            f"Decompressing {raw_path} to {formatted_path}"
                        )
                        with gzip.open(raw_path, "rb") as f_in:
                            with open(formatted_path, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        os.remove(raw_path)
                else:
                    raise ValueError(
                        f"Unsupported compression format: {dataset_conf.compressed.format}"
                    )

            for prompt_version, prompt in dataset_conf.prompts.items():
                for subset in dataset_conf.subsets:
                    if dataset_name == "mmlu":
                        dataset_f = format_mmlu(
                            f"{config.base.datasets_dir}/"
                            f"{config.format_datasets.raw_dir_path}/"
                            f"{dataset_name}/{subset}",
                            prompt,
                            dataset_conf.answer_map,
                        )
                    elif dataset_name == "gsm8k":
                        dataset_f = format_gsm8k(
                            f"{config.base.datasets_dir}/"
                            f"{config.format_datasets.raw_dir_path}/"
                            f"{dataset_name}/{subset}",
                            prompt.text,
                            generation_delimiter=prompt.generation_delimiter,
                        )
                    elif dataset_name == "notable_people":
                        dataset_f = format_notable(
                            f"{config.base.datasets_dir}/"
                            f"{config.format_datasets.raw_dir_path}/"
                            f"{dataset_name}/{subset}/"
                            f"{dataset_conf.compressed.target[:-3]}",
                            prompt,
                            dataset_conf.col_map,
                            question_tpl=dataset_conf.question_tpl,
                            filter=dataset_conf.filter,
                        )
                    else:
                        dataset_f = format_generic(
                            f"{config.base.datasets_dir}/"
                            f"{config.format_datasets.raw_dir_path}/"
                            f"{dataset_name}/{subset}/"
                            f"{subset}.{dataset_conf.format}",
                            prompt,
                            dataset_conf.col_map,
                            format=dataset_conf.format,
                        )

                    formatted_dir = (
                        f"{config.base.datasets_dir}/"
                        f"{config.format_datasets.formatted_dir_path}/"
                        f"{model_id}/{dataset_name}/{prompt_version}"
                    )
                    formatted_path = f"{formatted_dir}/{subset}.csv"

                    if not os.path.exists(formatted_dir):
                        logger.info(
                            f"Directory {formatted_dir} does not exist. Creating it."
                        )
                        os.makedirs(formatted_dir)

                    logger.info(f"Saving formatted dataset to {formatted_path}")

                    logger.info(
                        "Prompt templates variables substitution ({eos_token}})"
                    )

                    dataset_f = dataset_f.map(
                        lambda x: (
                            x.replace("{eos_token}", model_config.eos_token)
                            if isinstance(x, str)
                            else x
                        )
                    )

                    dataset_f.to_csv(
                        formatted_path,
                        index=False,
                    )


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument(
        "--model",
        dest="model",
        nargs="+",
        help=(
            "Model ID(s) to use for formatting datasets. "
            "If omitted, all models in params.generate_answers.models are used."
        ),
    )

    args = args_parser.parse_args()
    format_dataset(args.config, args.model)
