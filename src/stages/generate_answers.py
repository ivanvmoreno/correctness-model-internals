import argparse
import os
import shutil

import pandas as pd
from tqdm import tqdm

from src.data import load_statements
from src.model import (
    generate_const_hf,
    generate_const_vllm,
    generate_unconst_hf,
    generate_unconst_vllm,
    load_hf_model,
    load_vllm_model,
)
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.utils import sample_list_first_n, sample_list_random


def generate_answers(
    config_path: str,
    model_id: str,
    batch_size: int = 25,
) -> None:
    """Format datasets for question-answering tasks

    Args:
        config_path (str): Path to configuration file
        model_id (str): Model to use for generation
        batch_size (int): Batch size for generation
    """
    config = load_config(config_path)
    model_config = config.models[model_id]
    logger = get_logger("GENERATE_ANSWERS", config.base.log_level)

    logger.info(f"Generating answers for model {model_id}")

    if config.generate_answers.inference_engine == "hf":
        device = getattr(config.generate_answers, "device", "cuda")
        logger.info(f"Loading model onto GPU (device={device})")
        tokenizer, model = load_hf_model(
            config.base.models_dir,
            model_config.dir_path,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # IMPORTANT: left padded, as we're generating sequences in batches, and models are decoder-only
        generator_unconst = (
            lambda prompts, max_new_tokens, stop_words: generate_unconst_hf(
                tokenizer, model, prompts, max_new_tokens, stop_words
            )
        )
        generator_const = lambda prompts, choices_ids: generate_const_hf(
            tokenizer, model, prompts, choices_ids
        )
    elif config.generate_answers.inference_engine == "vllm":
        tokenizer, model = load_vllm_model(
            config.base.models_dir,
            model_config.dir_path,
            model_config.max_length,
        )
        generator_unconst = (
            lambda prompts, max_new_tokens, stop_words: generate_unconst_vllm(
                model, prompts, max_new_tokens, stop_words
            )
        )
        generator_const = lambda prompts, choices_ids: generate_const_vllm(
            model, prompts, choices_ids
        )
    else:
        raise ValueError(
            f"Unknown inference engine: {config.generate_answers.inference_engine}"
        )
    logger.info(
        f"Using inference engine: {config.generate_answers.inference_engine}"
    )

    for dataset_name, dataset_conf in config.datasets.items():
        if dataset_name == "mmlu":
            # Allow for space-aware variations of the answer map
            choices_ids = dataset_conf.answer_map + [
                f" {c}" for c in dataset_conf.answer_map
            ]
            if config.generate_answers.inference_engine == "hf":
                choices_ids = [
                    tokenizer.encode(choice, add_special_tokens=False)[0]
                    for choice in choices_ids
                ]
        for prompt_version, _ in dataset_conf.prompts.items():
            save_dir = os.path.join(
                config.base.generations_dir,
                model_id,
                dataset_name,
                prompt_version,
            )

            if os.path.exists(save_dir):
                logger.info(
                    f"Directory {save_dir} exists. Clearing previous generations."
                )
                shutil.rmtree(save_dir, ignore_errors=True)
            else:
                logger.info(f"Directory {save_dir} does not exist. Creating.")
            os.makedirs(save_dir)
            for subset in dataset_conf.subsets:
                formatted_path = os.path.join(
                    config.base.datasets_dir,
                    config.format_datasets.formatted_dir_path,
                    model_id,
                    dataset_name,
                    prompt_version,
                )
                if not os.path.exists(formatted_path):
                    raise FileNotFoundError(
                        f"Formatted file not found: {formatted_path}"
                    )
                logger.info(f"Loading statements from {formatted_path}")
                statements = load_statements(
                    os.path.join(formatted_path, f"{subset}.csv")
                )

                if (
                    config.generate_answers.max_dataset_size
                    and config.generate_answers.max_dataset_size
                    < len(statements)
                ):
                    logger.info(
                        f"Sampling {config.generate_answers.max_dataset_size} examples from dataset `{dataset_name}`, subset `{subset}` with strategy `{config.generate_answers.sample_strategy}`"
                    )
                    if config.generate_answers.sample_strategy == "first_n":
                        statements = sample_list_first_n(
                            statements, config.generate_answers.max_dataset_size
                        )
                    elif config.generate_answers.sample_strategy == "random":
                        statements = sample_list_random(
                            statements, config.generate_answers.max_dataset_size
                        )
                    else:
                        raise ValueError(
                            f"Sample strategy {config.generate_answers.sample_strategy} not supported"
                        )
                    # Store the sampled subset in a separate file
                    sampled_file = os.path.join(
                        formatted_path, f"{subset}_sampled.csv"
                    )
                    logger.info(f"Saving sampled subset to {sampled_file}")
                    pd.DataFrame(
                        statements, columns=["prompt", "answer"]
                    ).to_csv(sampled_file, index=False)

                # We only need the prompts for generation
                statements = [s[0] for s in statements]
                total_batches = (len(statements) + batch_size - 1) // batch_size
                batch_iter = tqdm(
                    range(0, len(statements), batch_size),
                    desc=f"Processing {subset}",
                    total=total_batches,
                    unit="batch",
                )

                for idx in batch_iter:
                    chunk = statements[idx : idx + batch_size]
                    save_file = os.path.join(
                        save_dir, f"{subset}_generations_{idx}.csv"
                    )
                    generations_data = []

                    if dataset_conf.answer_type == "multiple_choice":
                        const_answers = generator_const(chunk, choices_ids)
                        for prompt, answer in zip(chunk, const_answers):
                            generations_data.append(
                                {"prompt": prompt, "answer": answer}
                            )

                    elif dataset_conf.answer_type == "open_ended":

                        logger.info(
                            "Stop words template variables substitution ({eos_token}})"
                        )
                        stop_words = [
                            w.replace("{eos_token}", model_config.eos_token)
                            for w in dataset_conf.stop_words
                        ]

                        generations = generator_unconst(
                            chunk,
                            max_new_tokens=dataset_conf.max_new_tokens,
                            stop_words=stop_words,
                        )
                        for prompt, gen in zip(chunk, generations):
                            generations_data.append(
                                {"prompt": prompt, "answer": gen}
                            )

                    batch_iter.set_postfix_str(f"Saving {len(chunk)} samples")
                    pd.DataFrame(generations_data).to_csv(
                        save_file, index=False
                    )
                    logger.info(f"Saved generation results to {save_file}")

                logger.info(f"Joining all partial generation files")
                partial_files = [
                    f
                    for f in os.listdir(save_dir)
                    if f.endswith(".csv")
                    and f.startswith(f"{subset}_generations_")
                ]
                partial_files.sort(
                    key=lambda f: int(f.split("_")[-1].replace(".csv", ""))
                )

                joined_file = os.path.join(
                    save_dir, f"{subset}_generations.csv"
                )

                # Load each partial CSV and concatenate
                df_list = []
                for f in partial_files:
                    df_list.append(pd.read_csv(os.path.join(save_dir, f)))
                df_joined = pd.concat(df_list, ignore_index=True)
                df_joined.to_csv(joined_file, index=False)

                logger.info(f"Joined all partial files into {joined_file}")

                logger.info(f"Removing partial files")
                for f in partial_files:
                    os.remove(os.path.join(save_dir, f))
                logger.info(f"Removed partial files")


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument(
        "--model",
        dest="model",
        required=True,
        nargs="+",
        help="Model ID(s) to use for generation. Can be single or multiple models.",
    )
    args_parser.add_argument(
        "--batch-size", dest="batch_size", default=25, type=int
    )
    args = args_parser.parse_args()

    # Handle both single model and multiple models
    models = args.model if isinstance(args.model, list) else [args.model]

    # Run generation for each model
    for model_id in models:
        generate_answers(args.config, model_id, args.batch_size)
