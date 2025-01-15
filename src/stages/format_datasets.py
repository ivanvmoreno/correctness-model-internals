import os
import argparse

import pandas as pd

from src.utils.config import load_config
from src.utils.logging import get_logger


def format_multi_prompt(
    question, answers, sys_prompt: str, choices=["A", "B", "C", "D"]
) -> str:
    formatted_question = (
        f"{question}\n"
        + "\n".join([f"{choice}. {answer}" for choice, answer in zip(choices, answers)])
        + "\nAnswer:"
    )
    return f"{sys_prompt}\n{formatted_question}"


def format_open_prompt(question, sys_prompt: str) -> str:
    return f"{sys_prompt}\n{question}\nAnswer:"


def format_mmlu(
    path: str,
    sys_prompt: str,
    answer_map: list[str],
) -> pd.DataFrame:
    dataset_df = pd.read_parquet(path)
    prompts = dataset_df.apply(
        lambda row: format_multi_prompt(
            row["question"],
            row["choices"],
            sys_prompt,
        ),
        axis=1,
    )
    answers = dataset_df["answer"].apply(lambda a: answer_map[a])
    formatted = pd.DataFrame(
        {
            "prompt": prompts,
            "answer": answers,
            "subject": dataset_df["subject"],
        }
    )
    return formatted


def format_gsm8k(
    path: str,
    sys_prompt: str,
    answer_delim="####",
) -> pd.DataFrame:
    dataset_df = pd.read_parquet(path)
    prompts = dataset_df.apply(
        lambda row: format_open_prompt(
            row["question"],
            sys_prompt,
        ),
        axis=1,
    )
    dataset_df["parsed_answers"] = dataset_df["answer"].apply(
        lambda a: a.split(answer_delim)[-1].strip()
    )
    formatted = pd.DataFrame(
        {
            "prompt": prompts,
            "answer": dataset_df["answer"],
        }
    )
    return formatted


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
