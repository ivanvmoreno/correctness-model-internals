import os
import shutil

import pandas as pd
import os
import argparse

import pandas as pd
from tqdm import tqdm

from src.data import load_statements
from src.utils.utils import sample_list_random, sample_list_first_n
from src.model import load_model, generate_const, generate_unconst
from src.utils.config import load_config
from src.utils.logging import get_logger


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
                    config.format_dataset.dir_path,
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
                    and config.generate_answers.max_dataset_size < len(statements)
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
                    sampled_file = os.path.join(formatted_path, f"{subset}_sampled.csv")
                    logger.info(f"Saving sampled subset to {sampled_file}")
                    pd.DataFrame(statements, columns=["prompt", "answer"]).to_csv(
                        sampled_file, index=False
                    )
                # We only need the prompts for generation
                statements = [s[0] for s in statements]
                for idx in range(0, len(statements), min(batch_size, len(statements))):
                    chunk = statements[idx : idx + batch_size]
                    save_file = os.path.join(
                        save_dir, f"{subset}_generations_{idx}.csv"
                    )
                    generations_data = []
                    for statement in tqdm(chunk):
                        if dataset_name == "mmlu":
                            const_answer, _ = generate_const(
                                tokenizer, model, statement, choices_ids
                            )
                            generations_data.append(
                                {
                                    "prompt": statement,
                                    "answer": const_answer,
                                }
                            )

                        elif dataset_name == "gsm8k":
                            generation = generate_unconst(
                                tokenizer,
                                model,
                                statement,
                                max_new_tokens=config.generate_answers.max_new_tokens,
                                stop_word=config.generate_answers.stop_word,
                            )
                            generations_data.append(
                                {
                                    "prompt": statement,
                                    "answer": generation,
                                }
                            )

                    generations_df = pd.DataFrame(generations_data)
                    generations_df.to_csv(save_file, index=False)
                    logger.info(f"Saved generation results to {save_file}")

                logger.info(f"Joining all partial generation files")
                partial_files = [
                    f
                    for f in os.listdir(save_dir)
                    if f.endswith(".csv") and f.startswith(f"{subset}_generations_")
                ]
                partial_files.sort(
                    key=lambda f: int(f.split("_")[-1].replace(".csv", ""))
                )

                joined_file = os.path.join(save_dir, f"{subset}_generations.csv")

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
    args_parser.add_argument("--model", dest="model", required=True)
    args_parser.add_argument("--batch-size", dest="batch_size", default=25, type=int)
    args = args_parser.parse_args()
    generate_answers(args.config, args.model, args.batch_size)
