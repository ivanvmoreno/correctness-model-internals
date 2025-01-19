import argparse
import os

from src.data import format_gsm8k, format_mmlu
from src.utils.config import load_config
from src.utils.logging import get_logger


def format_dataset(
    config_path: str,
) -> None:
    """Format datasets for question-answering tasks

    Args:
        config_path (str): Path to configuration file
    """
    config = load_config(config_path)
    logger = get_logger("DATA_PREPROCESS", config.base.log_level)

    logger.info(f"Loading datasets from local path {config.base.datasets_dir}")

    for dataset_name, dataset_conf in config.datasets.items():
        for prompt_version, prompt in dataset_conf.prompts.items():
            for subset in dataset_conf.subsets:
                if dataset_name == "mmlu":
                    dataset_f = format_mmlu(
                        f"{config.base.datasets_dir}/{dataset_name}/{subset}",
                        prompt,
                        dataset_conf.answer_map,
                    )
                elif dataset_name == "gsm8k":
                    dataset_f = format_gsm8k(
                        f"{config.base.datasets_dir}/{dataset_name}/{subset}",
                        prompt,
                    )
                else:
                    raise ValueError(f"Dataset {dataset_name} not supported")
                formatted_dir = f"{config.base.datasets_dir}/{config.format_dataset.dir_path}/{dataset_name}/{prompt_version}"
                formatted_path = f"{formatted_dir}/{subset}.csv"

                if not os.path.exists(formatted_dir):
                    logger.info(
                        f"Directory {formatted_dir} does not exist. Creating it."
                    )
                    os.makedirs(formatted_dir)

                logger.info(f"Saving formatted dataset to {formatted_path}")

                dataset_f.to_csv(
                    formatted_path,
                    index=False,
                )


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    format_dataset(args.config)
