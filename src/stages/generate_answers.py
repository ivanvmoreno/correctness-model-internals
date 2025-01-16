import os
import shutil
from typing import Optional

import pandas as pd
import os
import argparse

import pandas as pd
from tqdm import tqdm

from src.data import load_statements
from src.utils.utils import sample_list
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
                config.base.datasets_dir,
                "generations",
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
                    f"{subset}.csv",
                )
                if not os.path.exists(formatted_path):
                    raise FileNotFoundError(
                        f"Formatted file not found: {formatted_path}"
                    )
                logger.info(f"Loading statements from {formatted_path}")
                statements = load_statements(formatted_path)
                if (
                    config.generate_answers.max_dataset_size
                    and config.generate_answers.max_dataset_size < len(statements)
                ):
                    logger.info(
                        f"Sampling {config.generate_answers.max_dataset_size} examples from dataset {dataset_name}, subset {subset}"
                    )
                    statements = sample_list(
                        statements, config.generate_answers.max_dataset_size
                    )
                for idx in range(0, len(statements), min(batch_size, len(statements))):
                    chunk = statements[idx : idx + batch_size]
                    save_file = os.path.join(
                        save_dir, f"{subset}_generations_{idx}.csv"
                    )
                    generations_data = []
                    for statement in tqdm(chunk):
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
