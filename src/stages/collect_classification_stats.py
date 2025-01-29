import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch as pt

from src.classifying import (
    ActivationsHandler,
    combine_activations_handlers,
    get_correctness_direction_classifier,
    get_logistic_regression_classifier,
)
from src.model import load_model
from src.utils.config import load_config
from src.utils.logging import get_logger


def load_activations(
    model_id,
    dataset_id,
    prompt_id,
    subset_id,
    input_type,
    layer,
    batch_ids=None,
):
    if batch_ids:
        batch_ids = [int(batch_id) for batch_id in batch_ids]

    paths = sorted(
        list(
            Path(
                f"./activations/{model_id}/{dataset_id}/{prompt_id}/{subset_id}/{input_type}"
            ).iterdir()
        ),
        key=lambda p: int(p.stem.split("_")[-1]),
    )

    activations_list, indices = [], []
    batch_size = None
    for batch_file in paths:
        batch_id = int(batch_file.stem.split("_")[-1])
        if batch_ids and batch_id not in batch_ids:
            continue

        activations = pt.load(batch_file, map_location=pt.device("cpu"))[layer]
        activations_list.append(activations)

        batch_size = activations.shape[0]

        if batch_size is None:
            batch_size = activations.shape[0]
        else:
            assert batch_size == activations.shape[0]

        indices.append(
            pd.Series(range(batch_size), name="index") + batch_id * batch_size
        )
    return (
        pt.cat(activations_list, dim=0),
        pd.concat(indices).reset_index(drop=True),
    )


def load_labels(model_id, dataset_id, prompt_id, subset_id, indices=None):
    paths = list(
        Path(f"./evaluations/{model_id}/{dataset_id}/{prompt_id}/").iterdir()
    )
    for path in paths:
        filename = path.stem
        if subset_id != filename.split("_generations_evaluated")[0]:
            continue
        df = pd.read_csv(path)
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)
        return df
    raise ValueError(
        f"No labels found for {model_id} {dataset_id} {prompt_id} {subset_id}"
    )


def classifier_experiment_run(
    run_configs,
    check_correctness_direction_classifier: bool = True,
    check_logistic_regression_classifier: bool = True,
    sample_equally=True,
    notes: str = "",
) -> None:

    overall_stats_dict = {
        "check_correctness_classifier": check_correctness_direction_classifier,
        "check_logistic_regression_classifier": check_logistic_regression_classifier,
        "timestamp": str(datetime.now()),
        "notes": notes,
        "results": [],
    }

    for run_config in run_configs:
        activations, indices = load_activations(
            model_id=run_config["model_id"],
            dataset_id=run_config["dataset_id"],
            prompt_id=run_config["prompt_id"],
            subset_id=run_config["subset_id"],
            input_type=run_config["input_type"],
            layer=run_config["layer"],
        )
        labels_df = load_labels(
            model_id=run_config["model_id"],
            dataset_id=run_config["dataset_id"],
            prompt_id=run_config["prompt_id"],
            subset_id=run_config["subset_id"],
            indices=indices,
        )

        if (
            not check_correctness_direction_classifier
            and not check_logistic_regression_classifier
        ):
            raise ValueError(
                "At least one of check_correctness_classifier or check_logistic_regression_classifier must be True"
            )

        activation_handler = ActivationsHandler(
            activations=activations, labels=labels_df["correct"].astype(bool)
        )
        if sample_equally:
            activation_handler = (
                activation_handler.sample_equally_across_groups(
                    group_labels=[False, True]
                )
            )

        activations_handler_folds = list(
            activation_handler.split_dataset(split_sizes=[0.2] * 5)
        )

        fold_stats = {}
        for i, activations_handler_test in enumerate(activations_handler_folds):
            activations_handler_train = combine_activations_handlers(
                [ah for j, ah in enumerate(activations_handler_folds) if j != i]
            )
            stats_dict = {
                "n_train": activations_handler_train.activations.shape[0],
                "n_test": activations_handler_test.activations.shape[0],
            }

            if check_correctness_direction_classifier:
                direction_classifier, direction_calculator = (
                    get_correctness_direction_classifier(
                        activations_handler_train=activations_handler_train,
                        activations_handler_test=activations_handler_test,
                    )
                )
                stats_dict["correctness_direction_classifier"] = (
                    direction_classifier.classification_metrics
                )
                stats_dict["activation_space_directions"] = {
                    name: getattr(direction_calculator, name).tolist()
                    for name in [
                        "classifying_direction",
                        "mean_activations",
                        "centroid_from",
                        "centroid_to",
                        "max_activations_from",
                        "min_activations_from",
                        "max_activations_to",
                        "min_activations_to",
                    ]
                }

            if check_logistic_regression_classifier:
                stats_dict["logistic_regression_classifier"] = (
                    get_logistic_regression_classifier(
                        activations_handler_train=activations_handler_train,
                        activations_handler_test=activations_handler_test,
                    )[0].classification_metrics
                )
            fold_stats[f"fold_{i}"] = stats_dict

        overall_stats_dict["results"].append(
            {
                **run_config,
                **fold_stats,
            }
        )
    return overall_stats_dict


def collect_classification_stats(
    config_path: str,
    model_id: str,
    layers: list[int],
) -> None:
    config = load_config(config_path)
    logger = get_logger("COLLECT_CLASSIFICATION_STATS", config.base.log_level)

    for dataset_name, dataset_conf in config.datasets.items():
        for prompt_version, _ in dataset_conf.prompts.items():
            for subset in dataset_conf.subsets:
                for input_type in config.capture_activations.input_type:
                    if isinstance(layers, int):
                        layers = [layers]
                    layers = [int(layer) for layer in layers]

                    # Capture activations for all layers
                    if layers == [-1]:
                        # Check directory for layer activations
                        layers = [
                            int(layer_dir.stem.split("_")[-1])
                            for layer_dir in list(
                                Path(
                                    f"{config.base.activations_dir}/{model_id}/{dataset_name}/{prompt_version}/{subset}/{input_type}"
                                ).iterdir()
                            )
                        ]

                    logger.info(f"Extracting activations from layers {layers}")

                    for l in layers:
                        save_dir = os.path.join(
                            config.base.classification_stats_dir,
                            model_id,
                            dataset_name,
                            prompt_version,
                            subset,
                            input_type,
                            f"layer_{l}",
                        )

                        if os.path.exists(save_dir):
                            logger.info(
                                f"Directory {save_dir} exists. Clearing previous activations."
                            )
                            shutil.rmtree(save_dir, ignore_errors=True)
                        else:
                            logger.info(
                                f"Directory {save_dir} does not exist. Creating."
                            )
                        os.makedirs(save_dir, exist_ok=True)

                        experiment_id = f"{model_id}_{dataset_name}_{prompt_version}_{subset}_{input_type}_{l}"

                        logger.info(f"Running experiment {experiment_id}")

                        res = classifier_experiment_run(
                            run_configs=[
                                {
                                    "model_id": model_id,
                                    "dataset_id": dataset_name,
                                    "prompt_id": prompt_version,
                                    "subset_id": subset,
                                    "input_type": input_type,
                                    "layer": l,
                                },
                            ],
                            check_correctness_direction_classifier=True,
                            check_logistic_regression_classifier=True,
                            sample_equally=True,
                        )

                        with open(
                            f"{save_dir}/classification_stats.json", "w"
                        ) as f:
                            json.dump(res, f, indent=4)
                        logger.info(
                            f"Saved classification stats for experiment {experiment_id} to {save_dir}/classification_stats.json"
                        )


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument("--model", dest="model", required=True)
    args_parser.add_argument(
        "--layers", dest="layers", nargs="+", default=[-1], type=int
    )
    args = args_parser.parse_args()
    collect_classification_stats(args.config, args.model, args.layers)
