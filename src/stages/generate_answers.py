import os
from typing import Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse

import pandas as pd

from src.utils.config import load_config
from src.utils.logging import get_logger


def load_model(
    models_path: str, model_dir: str, device="cuda:0"
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load model from local weights.
    """
    model = (
        AutoModelForCausalLM.from_pretrained(f"{models_path}/{model_dir}")
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(f"{models_path}/{model_dir}")
    return tokenizer, model


def load_statements(dataset_path: str):
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(dataset_path)
    statements = dataset["prompt"].tolist()
    return statements


def generate_const_answer(
    tokenizer, model, prompt: str, choices_ids
) -> Tuple[str, str]:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        last_token_logits = logits[0, -1, :]  # Logits for the last token
        masked_logits = last_token_logits.clone()
        masked_logits[:] = float("-inf")
        masked_logits[choices_ids] = last_token_logits[choices_ids]
        top_const_token_id = torch.argmax(masked_logits).item()
        top_unconst_token_id = torch.argmax(last_token_logits).item()
        top_const_token = tokenizer.decode([top_const_token_id])
        top_unconst_token = tokenizer.decode([top_unconst_token_id])

    return top_const_token, top_unconst_token


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

    logger.info(f"Generating answers for model {model}")

    logger.info(f"Loading model into GPU")
    tokenizer, model = load_model(config.base.models_dir_path, model)

    for dataset_name, dataset_conf in config.datasets.items():
        for prompt_version, _ in config.format_dataset.prompts:
            if dataset_name == "mmlu":
                for subset in dataset_conf.subsets:
                    formatted_path = os.path.join(
                        config.base.formatted_datasets_dir_path,
                        dataset_name,
                        prompt_version,
                        f"{subset}.csv",
                    )

                    if not os.path.exists(formatted_path):
                        logger.warning(f"Formatted file not found: {formatted_path}")
                        continue

                    logger.info(f"Loading statements from {formatted_path}")
                    statements = load_statements(formatted_path)

                    save_dir = os.path.join(
                        config.base.datasets_dir_path,
                        "generations",
                        dataset_name,
                        prompt_version,
                    )
                    os.makedirs(save_dir, exist_ok=True)

                    save_file = os.path.join(save_dir, f"{subset}_generations.csv")

                    if os.path.exists(save_file):
                        logger.info(f"Removing existing generations file: {save_file}")
                        os.remove(save_file)

                    generations_data = []
                    for statement in statements:
                        const_answer, unconst_answer = generate_const_answer(
                            tokenizer, model, statement, dataset_conf.answer_map
                        )
                        generations_data.append(
                            {
                                "statement": statement,
                                "const_answer": const_answer,
                                "unconst_answer": unconst_answer,
                            }
                        )

                    generations_df = pd.DataFrame(generations_data)
                    generations_df.to_csv(save_file, index=False)

                    logger.info(f"Saved generation results to {save_file}")

    if __name__ == "__main__":

        args_parser = argparse.ArgumentParser()
        args_parser.add_argument("--config", dest="config", required=True)
        args_parser.add_argument("--model", dest="model", required=True)
        args = args_parser.parse_args()
        generate_answers(args.config, model)
