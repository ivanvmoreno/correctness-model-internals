from typing import Literal

import numpy as np
import pandas as pd
import torch


def load_statements(dataset_path: str) -> list[tuple[str, str]]:
    dataset = pd.read_csv(dataset_path)
    statements = dataset["prompt"].tolist()
    answers = dataset["answer"].tolist()
    return list(zip(statements, answers))


def load_activations(
    activations_path: str,
    src_device="cuda",
) -> np.ndarray:
    """Load activations from a given file containing a torch.Tensor."""
    activations = torch.load(activations_path)
    if src_device == "cuda":
        activations = activations.cpu()
    return activations.numpy()


def format_multi_prompt(
    question,
    answers,
    sys_prompt: str,
    choices=["A", "B", "C", "D"],
    generation_delimiter="Answer:",
) -> str:
    formatted_question = f"{question}\n" + "\n".join(
        [f"{choice}. {answer}" for choice, answer in zip(choices, answers)]
    )
    if generation_delimiter:
        formatted_question += f"\n{generation_delimiter}"
    return f"{sys_prompt.strip()} {formatted_question}".strip()


def format_open_prompt(
    question,
    sys_prompt: str,
    generation_delimiter="Answer:",
) -> str:
    formatted_question = f"{question}"
    if generation_delimiter:
        formatted_question += f"\n{generation_delimiter}"
    return f"{sys_prompt.strip()} {formatted_question}".strip()


def format_mmlu(
    path: str,
    sys_prompt: str,
    answer_map: list[str],
    generation_delimiter="Answer:",
) -> pd.DataFrame:
    dataset_df = pd.read_parquet(path)
    prompts = dataset_df.apply(
        lambda row: format_multi_prompt(
            row["question"],
            row["choices"],
            sys_prompt,
            generation_delimiter=generation_delimiter,
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
    cot_answer_delim="####",
    generation_delimiter="Answer:",
) -> pd.DataFrame:
    dataset_df = pd.read_parquet(path)
    prompts = dataset_df.apply(
        lambda row: format_open_prompt(
            row["question"],
            sys_prompt,
            generation_delimiter=generation_delimiter,
        ),
        axis=1,
    )
    dataset_df["parsed_answers"] = dataset_df["answer"].apply(
        lambda a: a.split(cot_answer_delim)[-1].strip()
    )
    formatted = pd.DataFrame(
        {
            "prompt": prompts,
            "answer": dataset_df["parsed_answers"],
        }
    )
    return formatted


def format_generic(
    path: str,
    sys_prompt: str,
    col_names: dict[str, str],
    generation_delimiter="Answer:",
    format: Literal["parquet", "csv"] = "parquet",
) -> pd.DataFrame:
    if format == "parquet":
        dataset_df = pd.read_parquet(path)
    elif format == "csv":
        dataset_df = pd.read_csv(path)
    else:
        raise ValueError(
            "Invalid dataset format specified. Must be 'parquet' or 'csv'"
        )
    reversed_dict = {v: k for k, v in col_names.items()}
    dataset_df = dataset_df[reversed_dict.keys()].rename(columns=reversed_dict)
    prompts = dataset_df.apply(
        lambda row: format_open_prompt(
            row["prompt"],
            sys_prompt,
            generation_delimiter=generation_delimiter,
        ),
        axis=1,
    )
    formatted = pd.DataFrame(
        {
            "prompt": prompts,
            "answer": dataset_df["answer"],
        }
    )
    return formatted
