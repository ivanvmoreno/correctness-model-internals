import argparse
import os
import re

import pandas as pd

from src.utils.metrics import EVAL_METRICS
from src.utils.config import load_config
from src.utils.logging import get_logger


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


def generate_answers(
    config_path: str,
    model: str,
) -> None:
    """Format datasets for question-answering tasks

    Args:
        config_path (str): Path to configuration file
        model (str): Model to use for generation
    """
    config = load_config(config_path)
    logger = get_logger("GENERATE_ANSWERS", config.base.log_level)

    logger.info(f"Evaluating answers from model {model}")

    for dataset_name, dataset_conf in config.datasets.items():
        logger.info(f"Evaluating answers for dataset {dataset_name}")

        for prompt_version, _ in dataset_conf.prompts.items():
            logger.info(f"Evaluating answers for prompt version {prompt_version}")
            for subset in dataset_conf.subsets:
                generations_path = os.path.join(
                    config.base.generations_dir,
                    model,
                    dataset_name,
                    prompt_version,
                    f"{subset}_generations.csv",
                )
                logger.info(f"Evaluting answers for subset '{subset}'")
                ground_truth_path = os.path.join(
                    config.base.datasets_dir,
                    config.format_dataset.dir_path,
                    dataset_name,
                    prompt_version,
                    (
                        f"{subset}_sampled.csv"
                        if config.generate_answers.sample_strategy
                        else f"{subset}.csv"
                    ),
                )
                logger.info(f"Loading ground truth from {ground_truth_path}")
                ground_truth_df = pd.read_csv(ground_truth_path)
                logger.info(f"Loading generations from {generations_path}")
                generations_df = pd.read_csv(generations_path)
                if dataset_name == "mmlu":
                    # Convert from string label to index
                    y_true = ground_truth_df["answer"].apply(label_to_index)
                    y_pred = generations_df["answer"].apply(label_to_index)
                    # Compute metrics
                    metrics = {
                        metric: EVAL_METRICS[metric](y_true, y_pred)
                        for metric in EVAL_METRICS
                    }
                    # Prepare evaluation dataframe
                    metrics_rows = []
                    for metric_name in EVAL_METRICS.keys():
                        metrics_rows.append(
                            {
                                "metric": metric_name,
                                "value": metrics[metric_name],
                            }
                        )
                elif dataset_name == "gsm8k":
                    # Extract answers from open-ended generations
                    y_true = ground_truth_df["answer"].astype(str)
                    y_pred = generations_df["answer"].apply(
                        extract_answer_open_ended, regex=dataset_conf.answer_regex
                    )
                    y_correct = y_true == y_pred
                    metrics_rows = [
                        {
                            "metric": "accuracy",
                            "value": y_correct.mean(),
                        }
                    ]

                metrics_df = pd.DataFrame(metrics_rows)
                metrics_csv_path = os.path.join(
                    os.path.dirname(generations_path),
                    f"{subset}_metrics.csv",
                )
                metrics_df.to_csv(metrics_csv_path, index=False)
                logger.info(f"Saved metrics to {metrics_csv_path}")


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument("--model", dest="model", required=True)
    args = args_parser.parse_args()
    generate_answers(args.config, args.model)
