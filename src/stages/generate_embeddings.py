import argparse
import os
import shutil

import torch
import litellm
import numpy as np
from tqdm import tqdm  # Import tqdm

from src.data import load_dataset
from src.utils.config import load_config
from src.utils.logging import get_logger


def generate_embeddings(
    config_path: str,
    embedding_model_id: str,
    batch_size: int = 25,
) -> None:
    """Format datasets for question-answering tasks by generating embeddings with litellm.

    Args:
        config_path (str): Path to configuration file
        embedding_model_id (str): Embedding model ID to use with litellm
        batch_size (int, optional): Batch size for generation. Defaults to 25.
    """
    config = load_config(config_path)
    logger = get_logger("GENERATE_EMBEDDINGS", config.base.log_level)

    logger.info(f"Generating embeddings for model {embedding_model_id}")

    for dataset_name, dataset_conf in tqdm(
        config.datasets.items(), desc="Processing Datasets"
    ):
        save_dir = os.path.join(
            config.base.embeddings_dir,
            embedding_model_id.replace("/", "_"),
            dataset_name,
        )
        if os.path.exists(save_dir):
            # Skip if the directory exists
            logger.info(f"Directory {save_dir} exists. Skipping generation.")
            continue
        else:
            logger.info(f"Directory {save_dir} does not exist. Creating.")
        os.makedirs(save_dir, exist_ok=True)

        for subset in tqdm(
            dataset_conf.subsets,
            desc=f"Subsets for {dataset_name}",
            leave=False,
        ):
            datasets_path = os.path.join(
                config.base.datasets_dir,
                config.format_datasets.raw_dir_path,
                dataset_name,
                subset,
            )
            logger.info(f"Loading statements (inputs) from {datasets_path}")

            statements = (
                load_dataset(
                    datasets_path,
                    target_file=dataset_conf.embeddings_override_raw_path,
                )
                if dataset_conf.get("embeddings_override_raw_path", None)
                else load_dataset(datasets_path)
            )
            statements = statements["question"].tolist()

            num_statements = len(statements)
            total_batches = (num_statements + batch_size - 1) // batch_size

            logger.info(f"Generating embeddings for {dataset_name} - {subset}")
            batch_indices = range(0, num_statements, batch_size)
            for idx in tqdm(
                batch_indices,
                desc=f"Generating Embeddings ({subset})",
                total=total_batches,
                leave=False,
            ):
                chunk = statements[idx : idx + batch_size]
                try:
                    response = litellm.embedding(
                        model=embedding_model_id,
                        input=chunk,
                    )
                    batch_embeddings = [
                        item["embedding"] for item in response.data
                    ]
                    embeddings_tensor = torch.tensor(
                        np.array(batch_embeddings), dtype=torch.float32
                    )
                    torch.save(
                        embeddings_tensor,
                        os.path.join(
                            save_dir, f"{subset}_embeddings_batch_{idx}.pt"
                        ),
                    )
                except Exception as e:
                    logger.error(
                        f"Error generating/saving embeddings for {dataset_name}/{subset} batch starting at index {idx}: {e}"
                    )

            logger.info(
                f"Joining embeddings for dataset {dataset_name}, subset {subset}"
            )
            all_embeddings = []

            for idx in tqdm(
                batch_indices,
                desc=f"Joining Batches ({subset})",
                total=total_batches,
                leave=False,
            ):
                batch_path = os.path.join(
                    save_dir, f"{subset}_embeddings_batch_{idx}.pt"
                )
                if os.path.exists(batch_path):
                    try:
                        batch_embeddings = torch.load(batch_path)
                        all_embeddings.append(batch_embeddings)
                        # Remove batch file after loading
                        os.remove(batch_path)
                    except Exception as e:
                        logger.error(
                            f"Error loading/deleting batch file {batch_path}: {e}"
                        )

            if not all_embeddings:
                logger.warning(
                    f"No embedding batches found or loaded for {dataset_name} - {subset}. Skipping final save."
                )
                continue

            all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
            final_save_path = os.path.join(save_dir, f"{subset}_embeddings.pt")
            torch.save(all_embeddings_tensor, final_save_path)
            logger.info(
                f"Saved all embeddings for dataset {dataset_name}, subset {subset} to {final_save_path}"
            )


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument(
        "--model",
        dest="model",
        required=True,
        nargs="+",
        help="Embedding Model ID(s) (from litellm) to use for generating embeddings. Can be single or multiple models.",
    )
    args_parser.add_argument(
        "--batch-size", dest="batch_size", default=25, type=int
    )
    args = args_parser.parse_args()

    embedding_models = (
        args.model if isinstance(args.model, list) else [args.model]
    )

    for emb_model_id in tqdm(embedding_models, desc="Processing Models"):
        generate_embeddings(args.config, emb_model_id, args.batch_size)
