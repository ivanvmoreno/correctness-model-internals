import os
from typing import Tuple, Optional

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse

import pandas as pd
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logging import get_logger


def load_model(
    models_path: str, model_dir: str, device="cuda:0"
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load model from local weights.
    """
    model = AutoModelForCausalLM.from_pretrained(f"{models_path}/{model_dir}").to(
        device
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(f"{models_path}/{model_dir}")
    return tokenizer, model


def load_statements(dataset_path: str):
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(dataset_path)
    statements = dataset["prompt"].tolist()
    return statements


def sample_list(l, n):
    """
    Sample n elements from list l.
    """
    return [l[i] for i in torch.randperm(len(l))[:n]]


def generate_const(tokenizer, model, prompt: str, choices_ids) -> Tuple[str, str]:
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


def generate_unconst(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 64,
    stop_token: Optional[str] = None,
) -> str:
    """
    Generates text using the Hugging Face built-in generate() method.
    This approach automatically caches past key/value states
    and is much faster than running a full forward pass per token.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # Custom EOS (stop) token
    if stop_token is not None:
        # Note: This only works if stop_token maps to exactly one token
        # For multi-token stop sequences, more logic is needed
        stop_token_id = tokenizer.encode(stop_token, add_special_tokens=False)
        if len(stop_token_id) == 1:
            eos_token_id = stop_token_id[0]
        else:
            eos_token_id = None  # fallback if multi-token
    else:
        eos_token_id = None

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=eos_token_id,
    )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if stop_token is not None and stop_token in generated_text:
        generated_text = generated_text.split(stop_token)[0]

    return generated_text


def generate_answers(
    config_path: str,
    model_id: str,
    batch_size: int = 25,
    max_dataset_size: Optional[int] = None,
) -> None:
    """Format datasets for question-answering tasks

    Args:
        config_path (str): Path to configuration file
        model_id (str): Model to use for generation
        batch_size (int): Batch size for generation
    """
    config = load_config(config_path)
    logger = get_logger("GENERATE_ANSWERS", config.base.log_level)

    logger.info(f"Generating answers for model {model_id}")

    logger.info(f"Loading model into GPU")
    tokenizer, model = load_model(
        config.base.models_dir, config.models[model_id].dir_path
    )

    for dataset_name, dataset_conf in config.datasets.items():
        if dataset_name == "mmlu":
            # Allow for space-aware variations of the answer map
            choices_ids = [
                tokenizer.encode(choice, add_special_tokens=False)[0]
                for choice in dataset_conf.answer_map
                + [f" {c}" for c in dataset_conf.answer_map]
            ]
        for prompt_version, _ in dataset_conf.prompts.items():
            save_dir = os.path.join(
                config.base.datasets_dir,
                "generations",
                model_id,
                dataset_name,
                prompt_version,
            )
            os.makedirs(save_dir, exist_ok=True)
            for subset in dataset_conf.subsets:
                formatted_path = os.path.join(
                    config.base.datasets_dir,
                    config.format_dataset.dir_path,
                    dataset_name,
                    prompt_version,
                    f"{subset}.csv",
                )
                if not os.path.exists(formatted_path):
                    raise FileNotFoundError(
                        f"Formatted file not found: {formatted_path}"
                    )
                logger.info(f"Loading statements from {formatted_path}")
                statements = load_statements(formatted_path)
                if (
                    config.format_dataset.max_dataset_size
                    and config.format_dataset.max_dataset_size < len(statements)
                ):
                    logger.info(
                        f"Sampling {max_dataset_size} examples from dataset {dataset_name}, subset {subset}"
                    )
                    statements = sample_list(statements, max_dataset_size)
                for idx in range(0, len(statements), batch_size):
                    chunk = statements[idx : idx + batch_size]
                    for statement in tqdm(chunk):
                        save_file = os.path.join(
                            save_dir, f"{subset}_generations_{idx}.csv"
                        )
                        generations_data = []

                        if dataset_name == "mmlu":
                            const_answer, unconst_answer = generate_const(
                                tokenizer, model, statement, choices_ids
                            )
                            generations_data.append(
                                {
                                    "statement": statement,
                                    "const_answer": const_answer,
                                    "unconst_answer": unconst_answer,
                                }
                            )
                        if dataset_name == "gsm8k":
                            for statement in chunk:
                                generation = generate_unconst(
                                    tokenizer, model, statement
                                )
                                generations_data.append(
                                    {
                                        "statement": statement,
                                        "answer": generation,
                                    }
                                )

                        generations_df = pd.DataFrame(generations_data)
                        generations_df.to_csv(save_file, index=False)
                        logger.info(f"Saved generation results to {save_file}")


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument("--model", dest="model", required=True)
    args_parser.add_argument("--batch-size", dest="batch_size", default=25)
    args = args_parser.parse_args()
    generate_answers(args.config, args.model, args.batch_size)
