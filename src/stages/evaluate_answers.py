import argparse
import json
import os
import re
import asyncio

import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm  # Import tqdm's asyncio version

from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.metrics import EVAL_METRICS, serialize_metrics

from src.llm_judge import evaluate_answer_llm, QPSRateLimiter


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
    """
    match = re.search(regex, generation)
    if match:
        return match.group()
    else:
        return None


def evaluate_answers(
    config_path: str,
    model: str,
    llm_judge: bool = False,
) -> None:
    config = load_config(config_path)
    logger = get_logger("EVALUATE_ANSWERS", config.base.log_level)

    logger.info(f"Evaluating answers from model {model}")

    if (
        llm_judge
        and config.evaluate_answers.llm_judge.inference_engine == "litellm"
    ):
        rate_limiter = QPSRateLimiter(
            requests_per_second=config.inference_engines.litellm.qps_limit,
        )
    else:
        rate_limiter = None

    for dataset_name, dataset_conf in config.datasets.items():
        logger.info(f"Evaluating answers for dataset {dataset_name}")

        for prompt_version, _ in dataset_conf.prompts.items():
            logger.info(
                f"Evaluating answers for prompt version {prompt_version}"
            )
            for subset in dataset_conf.subsets:
                logger.info(f"Evaluating answers for subset '{subset}'")
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
                    model,
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
                    logger.info(
                        f"Loading ground truth from {ground_truth_full}"
                    )
                    ground_truth_df = pd.read_csv(ground_truth_full)

                # Ensure alignment if sampling occurred
                if len(generations_df) != len(ground_truth_df):
                    logger.warning(
                        f"Mismatch in lengths between generations ({len(generations_df)}) and ground truth ({len(ground_truth_df)}) for subset '{subset}'. Aligning based on ground truth index."
                    )
                    ground_truth_df = ground_truth_df.set_index(
                        "original_index", drop=False
                    )  # Assuming 'original_index' exists from sampling
                    generations_df = generations_df.set_index(
                        "original_index", drop=False
                    )
                    common_indices = ground_truth_df.index.intersection(
                        generations_df.index
                    )
                    ground_truth_df = ground_truth_df.loc[common_indices]
                    generations_df = generations_df.loc[common_indices]
                    ground_truth_df = ground_truth_df.reset_index(drop=True)
                    generations_df = generations_df.reset_index(drop=True)

                if dataset_conf.eval_type == "constrained_tokens":
                    y_true = ground_truth_df["answer"].apply(label_to_index)
                    y_pred = generations_df["answer"].apply(label_to_index)
                    y_correct = y_true == y_pred
                    metrics = {
                        metric: EVAL_METRICS[metric](y_true, y_pred)
                        for metric in EVAL_METRICS
                    }
                elif dataset_conf.eval_type == "regex_match":
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
                    ).astype(bool)
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
                    y_correct = pd.Series(
                        [
                            s.lower()
                            in [
                                item.lower()
                                for item in re.findall(
                                    r"""['"]([^'"]+)['"]""", lst
                                )
                            ]
                            for s, lst in zip(y_pred, y_true)
                        ]
                    )
                    metrics = {
                        "accuracy": y_correct.mean(),
                    }
                else:
                    raise ValueError(
                        f"Unknown eval_type: {dataset_conf.eval_type}"
                    )

                logger.info("Checking for IDK responses in incorrect answers")
                generations_df["idk_response"] = False
                incorrect_mask = ~y_correct

                # Use .loc for potentially non-contiguous indices after alignment
                for idx in generations_df.loc[incorrect_mask].index:
                    answer = str(generations_df.loc[idx, "answer"]).lower()
                    is_idk = any(
                        pattern.lower() in answer
                        for pattern in config.evaluate_answers.idk_match
                    )
                    if is_idk:
                        generations_df.loc[idx, "idk_response"] = True

                metrics["idk_rate"] = generations_df["idk_response"].mean()

                y_correct = pd.Series(
                    y_correct, index=generations_df.index
                )  # Ensure index alignment

                llm_judge_used_and_applicable = (
                    llm_judge
                    and dataset_name
                    in config.evaluate_answers.llm_judge.datasets
                )

                if llm_judge_used_and_applicable:
                    logger.info(
                        f"Evaluating incorrect answers with LLM judge for subset '{subset}'"
                    )
                    original_y_correct = y_correct.copy()
                    incorrect_mask_for_judge = (
                        ~y_correct & ~generations_df["idk_response"]
                    )
                    incorrect_indices = list(
                        generations_df[incorrect_mask_for_judge].index
                    )

                    async def evaluate_failed():
                        tasks = []
                        for idx in incorrect_indices:
                            # Ensure 'original_statement' exists and handle potential missing keys gracefully
                            original_statement = ground_truth_df.loc[idx].get(
                                "original_statement", "Statement not available"
                            )
                            generated_answer = generations_df.loc[idx, "answer"]
                            ground_truth_answer = y_true[idx]
                            task = evaluate_answer_llm(
                                evaluator_system=config.evaluate_answers.llm_judge.prompt.system,
                                evaluator_user=config.evaluate_answers.llm_judge.prompt.user,
                                evaluator_model=config.evaluate_answers.llm_judge.model,
                                question=original_statement,
                                answer=generated_answer,
                                ground_truth=ground_truth_answer,
                                rate_limiter=rate_limiter,
                                na_value=config.evaluate_answers.idk_class_value,
                            )
                            tasks.append(task)
                        # Use tqdm.asyncio.gather to show progress
                        results = await async_tqdm.gather(
                            *tasks,
                            desc=f"LLM Judging {subset} ({len(tasks)} items)",
                            total=len(tasks),
                        )
                        return results

                    if incorrect_indices:
                        llm_results = asyncio.run(evaluate_failed())

                        for i, idx in enumerate(incorrect_indices):
                            try:
                                # Check if result is not None before converting to int
                                if llm_results[i] is not None:
                                    eval_value = int(llm_results[i])
                                    if eval_value:
                                        y_correct.loc[idx] = True
                                else:
                                    # Treat None result same as IDK or ValueError
                                    generations_df.loc[idx, "idk_response"] = (
                                        True
                                    )
                                    logger.warning(
                                        f"LLM Judge returned None for index {idx}. Marked as IDK."
                                    )

                            except (ValueError, TypeError) as e:
                                # IDK response or other error from LLM
                                generations_df.loc[idx, "idk_response"] = True
                                logger.warning(
                                    f"LLM Judge returned non-integer or error for index {idx}: '{llm_results[i]}' (Error: {e}). Marked as IDK."
                                )

                        llm_judge_correct = y_correct & ~original_y_correct

                        # Recompute metrics potentially affected by LLM judge
                        metrics["accuracy"] = y_correct.mean()
                        metrics["idk_rate"] = generations_df[
                            "idk_response"
                        ].mean()
                        metrics["total_llm_judge_corrections"] = (
                            llm_judge_correct.sum()
                        )
                        # Recalculate other metrics if they depend on y_correct
                        if dataset_conf.eval_type == "constrained_tokens":
                            metrics["f1"] = EVAL_METRICS["f1"](
                                y_true, y_correct.astype(int) * y_true.max()
                            )  # Assuming binary/multiclass where incorrect is 0
                            # Add recalculation for other relevant metrics like precision, recall if needed

                    else:
                        logger.info(
                            f"No incorrect (non-IDK) answers to evaluate with LLM judge for subset '{subset}'."
                        )
                        llm_judge_correct = pd.Series(
                            [False] * len(generations_df),
                            index=generations_df.index,
                        )
                        metrics["total_llm_judge_corrections"] = 0

                evaluations_path = os.path.join(
                    config.base.evaluations_dir,
                    model,
                    dataset_name,
                    prompt_version,
                )
                os.makedirs(evaluations_path, exist_ok=True)

                generations_df["ground_truth"] = y_true
                generations_df["correct"] = y_correct
                if llm_judge_used_and_applicable:
                    generations_df["original_correct"] = original_y_correct
                    generations_df["llm_judge_correct"] = llm_judge_correct

                generations_eval_csv_path = os.path.join(
                    evaluations_path, f"{subset}_generations_evaluated.csv"
                )
                generations_df.to_csv(generations_eval_csv_path, index=False)
                logger.info(
                    f"Saved evaluated generations to {generations_eval_csv_path}"
                )

                logger.info("Post-processing evaluation metrics")
                metrics = serialize_metrics(metrics)
                metrics_path = os.path.join(
                    evaluations_path, f"{subset}_metrics.json"
                )
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=4)
                logger.info(f"Saved metrics to {metrics_path}")

    all_metrics = {}
    for dataset_name, dataset_conf in config.datasets.items():
        all_metrics[dataset_name] = {}
        for prompt_version, _ in dataset_conf.prompts.items():
            all_metrics[dataset_name][prompt_version] = {}
            for subset in dataset_conf.subsets:
                metrics_path = os.path.join(
                    config.base.evaluations_dir,
                    model,
                    dataset_name,
                    prompt_version,
                    f"{subset}_metrics.json",
                )
                try:
                    with open(metrics_path, "r") as f:
                        all_metrics[dataset_name][prompt_version][subset] = (
                            json.load(f)
                        )
                except FileNotFoundError:
                    logger.error(
                        f"Metrics file not found: {metrics_path}. Skipping."
                    )

    metrics_joined_path = os.path.join(
        config.base.evaluations_dir, model, "metrics.json"
    )
    with open(metrics_joined_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    logger.info(f"Joined metrics saved to {metrics_joined_path}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument(
        "--model",
        dest="model",
        required=True,
        nargs="+",
        help="Model ID(s) to use for evaluation. Can be single or multiple models.",
    )
    args_parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Use LLM as judge for evaluation",
    )
    args = args_parser.parse_args()
    models = args.model if isinstance(args.model, list) else [args.model]

    for model_id in models:
        evaluate_answers(args.config, model_id, args.llm_judge)
