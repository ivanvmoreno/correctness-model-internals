from typing import Optional, Union, List
import argparse
import os

import pandas as pd

from src.utils.config import load_config
from src.utils.logging import get_logger


def add_noise(
    config_path: str,
    models: Optional[Union[str, List[str]]],
    datasets: Optional[Union[str, List[str]]],
    noise_percentage: float,
) -> None:
    """Add random noise to dataset evaluation labels by flipping.

    Args:
        config_path (str): Path to configuration file
        models (Optional[Union[str, List[str]]]): Model ID(s) to modify the datasets for.
            If None, all models in params.generate_answers.models are used.
        datasets (Optional[Union[str, List[str]]]): Dataset ID(s) to modify.
            If None, all datasets in params.datasets are used.
    """
    config = load_config(config_path)
    logger = get_logger("ADD_NOISE", config.base.log_level)

    logger.info(f"Loading datasets from local path {config.base.datasets_dir}")

    if models is None:
        logger.info(
            "No model argument provided. Using all models from params.generate_answers.models"
        )
        models = config.generate_answers.models

    if isinstance(models, str):
        models = [models]

    if datasets is None:
        logger.info(
            "No dataset argument provided. Using all datasets from params.datasets"
        )
        datasets = list(config.datasets.keys())

    if isinstance(datasets, str):
        datasets = [datasets]

    datasets = [(ds_name, config.datasets[ds_name]) for ds_name in datasets]

    for dataset_name, dataset_conf in datasets:
        logger.info(f"Altering dataset {dataset_name}")
        altered_indices = None

        for model_id in models:
            logger.info(f"Altering datasets for model {model_id}")
            for prompt_version, _ in dataset_conf.prompts.items():
                for subset in dataset_conf.subsets:
                    evals_dir = f"{config.base.evaluations_dir}/{model_id}/{dataset_name}/{prompt_version}"
                    evals_path = (
                        f"{evals_dir}/{subset}_generations_evaluated.csv"
                    )

                    if not os.path.exists(evals_path):
                        logger.warning(
                            f"Evaluated generations file not found at {evals_path}. "
                            "There is something wrong with this eval, or the current params.yaml "
                            "doesn't reflect the configuration used at eval time. Skipping."
                        )
                        continue

                    evals = pd.read_csv(evals_path)
                    evals["correct"] = evals["correct"].astype(int).astype(bool)
                    evals["correct_original"] = evals["correct"]

                    # Altered samples are consistent across models, prompt versions, and subsets
                    if altered_indices is None:
                        num_noisy = int(len(evals) * noise_percentage / 100)
                        logger.info(
                            f"Altering {num_noisy} samples out of {len(evals)} ({noise_percentage}%) for dataset {dataset_name}"
                        )
                        altered_indices = evals.sample(
                            n=num_noisy,
                            random_state=config.base.random_seed,
                        ).index

                    evals.loc[altered_indices, "correct"] = ~evals.loc[
                        altered_indices, "correct"
                    ]

                    logger.info(
                        f"Saving altered combination to {evals_path}. Renaming original to {evals_path}.original"
                    )
                    backup_path = f"{evals_path}.original"
                    if not os.path.exists(backup_path):
                        os.rename(evals_path, backup_path)
                    else:
                        logger.warning(
                            f"Backup already exists at {backup_path}, not overwriting."
                        )
                    evals.to_csv(evals_path, index=False)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument(
        "--datasets",
        dest="datasets",
        nargs="+",
        help="Dataset ID(s) to use for capturing activations. Can be single or multiple datasets.",
    )
    args_parser.add_argument(
        "--models",
        dest="models",
        nargs="+",
        help="Model ID(s) to use for capturing activations. Can be single or multiple models.",
    )
    args_parser.add_argument(
        "--noise-percentage",
        dest="noise_percentage",
        help="Amount of noise to add (labels to flip) to the dataset (as a percentage)",
        type=float,
        required=True,
    )
    args = args_parser.parse_args()
    add_noise(args.config, args.models, args.datasets, args.noise_percentage)
