import argparse
import json
import os
import re

import pandas as pd

from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.metrics import EVAL_METRICS


def label_to_index(label, answer_map=["A", "B", "C", "D"]):
    """
    Return the index of a valid label in answer_map.
    If the label is out of the map, return -1 (i.e., "incorrect").
    """
    label_stripped = str(label).strip()
    if label_stripped in answer_map:
        return answer_map.index(label_stripped)
    else:
        # Treat anything else as "incorrect."
        return -1


def extract_answer_open_ended(generation: str, regex: str):
    """
    Extract the answer from an open-ended generation.

    Args:
        generation (str): The generated text.
        regex (str): The regex to extract the answer.
        skip_prompt (bool): Whether regex is included in the prompt.
    """
    match = re.search(regex, generation)
    if match:
        return match.group()
    else:
        return None


def evaluate_answers(
    config_path: str,
    model: str,
) -> None:
    """Format datasets for question-answering tasks

    Args:
        config_path (str): Path to configuration file
        model (str): Model to use for generation
    """
    config = load_config(config_path)
    logger = get_logger("EVALUATE_ANSWERS", config.base.log_level)

    logger.info(f"Evaluating answers from model {model}")

    for dataset_name, dataset_conf in config.datasets.items():
        logger.info(f"Evaluating answers for dataset {dataset_name}")

        for prompt_version, _ in dataset_conf.prompts.items():
            logger.info(
                f"Evaluating answers for prompt version {prompt_version}"
            )
            for subset in dataset_conf.subsets:
                logger.info(f"Evaluting answers for subset '{subset}'")
                generations_path = os.path.join(
                    config.base.generations_dir,
                    model,
                    dataset_name,
                    prompt_version,
                    f"{subset}_generations.csv",
                )
                ground_truth_path = os.path.join(
                    config.base.datasets_dir,
                    config.format_datasets.formatted_dir_path,
                    dataset_name,
                    prompt_version,
                )
                logger.info(f"Loading generations from {generations_path}")
                generations_df = pd.read_csv(generations_path)
                if config.generate_answers.max_dataset_size:
                    try:
                        ground_truth_sampled = (
                            f"{ground_truth_path}/{subset}_sampled.csv"
                        )
                        logger.info(
                            f"Loading ground truth from {ground_truth_sampled}"
                        )
                        ground_truth_df = pd.read_csv(ground_truth_sampled)
                    except FileNotFoundError:
                        logger.warning(
                            f"Subset `{subset}` was not sampled. Loading full dataset."
                        )
                        ground_truth_full = f"{ground_truth_path}/{subset}.csv"
                        logger.info(
                            f"Loading ground truth from {ground_truth_full}"
                        )
                        ground_truth_df = pd.read_csv(ground_truth_full)
                else:
                    ground_truth_full = f"{ground_truth_path}/{subset}.csv"
                    logger.info(f"Loading ground truth from {ground_truth_full}")
                    ground_truth_df = pd.read_csv(ground_truth_full)
                if dataset_conf.eval_type == "constrained_tokens":
                    # Convert from string label to index
                    y_true = ground_truth_df["answer"].apply(label_to_index)
                    y_pred = generations_df["answer"].apply(label_to_index)
                    y_correct = y_true == y_pred
                    # Compute metrics
                    metrics = {
                        metric: EVAL_METRICS[metric](y_true, y_pred)
                        for metric in EVAL_METRICS
                    }
                elif dataset_conf.eval_type == "regex_match":
                    # Extract answers from open-ended generations
                    y_true = ground_truth_df["answer"].astype(str)
                    y_pred = generations_df["answer"].apply(
                        extract_answer_open_ended,
                        regex=dataset_conf.answer_regex,
                    )
                    y_correct = y_true == y_pred
                    metrics = {
                        "accuracy": y_correct.mean(),
                    }
                elif dataset_conf.eval_type == "answers_map":
                    answers_map = json.loads(
                        open(
                            os.path.join(
                                config.base.datasets_dir,
                                config.format_datasets.raw_dir_path,
                                dataset_name,
                                "eval_map.json",
                            ),
                        ).read()
                    )
                    y_true = ground_truth_df["answer"]
                    y_pred = generations_df["answer"]
                    y_correct = pd.Series(
                        [
                            x in answers_map[y_true[i]]
                            for i, x in zip(y_pred.index, y_pred)
                        ]
                    ).astype(int)
                    metrics = {
                        "accuracy": y_correct.mean(),
                    }
                elif dataset_conf.eval_type == "exact_match":
                    y_true = ground_truth_df["answer"].astype(str)
                    y_pred = generations_df["answer"].astype(str)
                    y_correct = y_true == y_pred
                    metrics = {
                        "accuracy": y_correct.mean(),
                    }
                elif dataset_conf.eval_type == "list_of_answers":
                    y_true = list(ground_truth_df["answer"])
                    y_pred = generations_df["answer"].astype(str)
                    y_correct = pd.Series([
                        int(s.lower() in [item.lower() for item in re.findall(r"""['"]([^'"]+)['"]""", lst)])
                        for s, lst in zip(y_pred, y_true)
                    ])
                    
                    metrics = {
                        "accuracy": y_correct.mean(),
                    }
                    
                    metrics = {
                        "accuracy": y_correct.mean(),
                    }
                    
                evaluations_path = os.path.join(
                    config.base.evaluations_dir,
                    model,
                    dataset_name,
                    prompt_version,
                )
                generations_eval_csv_path = os.path.join(
                    evaluations_path,
                    f"{subset}_generations_evaluated.csv",
                )
                os.makedirs(evaluations_path, exist_ok=True)
                generations_df["ground_truth"] = y_true
                generations_df["correct"] = y_correct
                generations_df.to_csv(
                    generations_eval_csv_path,
                    index=False,
                )
                logger.info(
                    f"Saved evaluated generations to {generations_eval_csv_path}"
                )
                metrics_path = os.path.join(
                    evaluations_path,
                    f"{subset}_metrics.json",
                )
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=4)
                logger.info(f"Saved metrics to {metrics_path}")

    # Join all metrics (dataset, prompt_version, subset) into a single file
    metrics = {}
    for dataset_name, dataset_conf in config.datasets.items():
        metrics[dataset_name] = {}
        for prompt_version, _ in dataset_conf.prompts.items():
            metrics[dataset_name][prompt_version] = {}
            for subset in dataset_conf.subsets:
                metrics_path = os.path.join(
                    config.base.evaluations_dir,
                    model,
                    dataset_name,
                    prompt_version,
                    f"{subset}_metrics.json",
                )
                with open(metrics_path, "r") as f:
                    metrics[dataset_name][prompt_version][subset] = json.load(f)
    metrics_joined_path = os.path.join(
        config.base.evaluations_dir,
        model,
        "metrics.json",
    )
    with open(metrics_joined_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Joined metrics saved to {metrics_joined_path}")


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument("--model", dest="model", required=True)
    args = args_parser.parse_args()
    evaluate_answers(args.config, args.model)
