import os
from typing import Literal

import numpy as np
import pandas as pd
import torch


def load_dataset(
    dataset_path: str,
    target_file: str = None,
) -> pd.DataFrame:
    """Load a dataset from a given path. Detects files in the path, and loads them as a pandas DataFrame."""
    files = [target_file] if target_file else os.listdir(dataset_path)
    dataset = pd.DataFrame()
    for file in files:
        if file.endswith(".csv"):
            try:
                dataset = pd.concat(
                    [dataset, pd.read_csv(os.path.join(dataset_path, file))]
                )
            except Exception as e:
                try:
                    dataset = pd.concat(
                        [
                            dataset,
                            pd.read_csv(
                                os.path.join(dataset_path, file),
                                low_memory=False,
                                encoding_errors="replace",
                                on_bad_lines="skip",
                            ).dropna(),
                        ]
                    )
                except Exception as e:
                    logger.warning("Failed to load %s: %s", file, e)
        elif file.endswith(".parquet"):
            dataset = pd.concat(
                [dataset, pd.read_parquet(os.path.join(dataset_path, file))]
            )
    return dataset


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
            "original_statement": dataset_df["prompt"],
            "prompt": prompts,
            "answer": answers,
            "subject": dataset_df["subject"],
        }
    )
    return formatted


def format_notable(
    path: str,
    sys_prompt: str,
    col_names: dict[str, str],
    generation_delimiter: str = "Answer:",
    question_tpl: str = "What year was {name} ({occupation}, {birthplace}) born?",
    filter: str = None,
) -> pd.DataFrame:
    reversed_dict = {v: k for k, v in col_names.items()}
    dataset_df = (
        pd.read_csv(
            path,
            low_memory=False,
            encoding_errors="replace",
            sep=",",
            quotechar="'",
            on_bad_lines="skip",
            usecols=reversed_dict.keys(),
        )
        .rename(columns=reversed_dict)
        .dropna()
    )
    if filter:
        dataset_df = dataset_df.query(filter)
    dataset_df.loc[:, "name"] = dataset_df["name"].str.replace(
        "_", " ", regex=False
    )
    dataset_df.loc[:, "occupation"] = dataset_df["occupation"].str.replace(
        "_", " ", regex=False
    )
    birthplace_pattern = r"(?:D:_')?([^']+)(?:'[^']*)?"
    dataset_df.loc[:, "birthplace"] = (
        dataset_df["birthplace"]
        .str.extract(birthplace_pattern, expand=False)
        .str.replace("_", " ", regex=False)
    )
    questions = dataset_df.apply(
        lambda row: question_tpl.format(
            name=row["name"],
            occupation=row["occupation"],
            birthplace=row["birthplace"],
        ),
        axis=1,
    )
    prompts = questions.apply(
        lambda q: format_open_prompt(
            q,
            sys_prompt,
            generation_delimiter=generation_delimiter,
        )
    )
    answers = dataset_df["birth_year"].astype(int).astype(str)
    formatted = pd.DataFrame(
        {
            "original_statement": questions,
            "prompt": prompts,
            "answer": answers,
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
            "original_statement": dataset_df["question"],
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
            "original_statement": dataset_df["prompt"],
            "prompt": prompts,
            "answer": dataset_df["answer"],
        }
    )
    return formatted
