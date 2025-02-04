import argparse
import os
import shutil

import torch

from src.data import load_statements
from src.model import get_acts, load_model
from src.utils.config import load_config
from src.utils.logging import get_logger


def capture_activations(
    config_path: str,
    model_id: str,
    layers: list,
    batch_size: int = 25,
    device: str = "cuda:0",
) -> None:
    """Format datasets for question-answering tasks by capturing model activations.

    Args:
        config_path (str): Path to configuration file
        model_id (str): Model to use for generation
        layers (list): List of layers to extract activations from
        batch_size (int, optional): Batch size for generation. Defaults to 25.
        device (str, optional): Device to run inference on. Defaults to 'cuda:0'.
    """
    config = load_config(config_path)
    logger = get_logger("GENERATE_ACTS", config.base.log_level)

    logger.info(f"Generating activations for model {model_id}")

    logger.info(f"Loading model into GPU (device={device})")
    tokenizer, model = load_model(
        config.base.models_dir, config.models[model_id].dir_path
    )
    model.to(device)

    # Capture activations for all layers
    if layers is None:
        layers = list(range(len(model.model.layers)))

    if isinstance(layers, int):
        layers = [layers]
    layers = [int(layer) for layer in layers]

    logger.info(f"Extracting activations from layers {layers}")

    for dataset_name, dataset_conf in config.datasets.items():
        for prompt_version, _ in dataset_conf.prompts.items():
            save_dir = os.path.join(
                config.base.activations_dir,
                model_id,
                dataset_name,
                prompt_version,
            )
            if os.path.exists(save_dir):
                logger.info(
                    f"Directory {save_dir} exists. Clearing previous activations."
                )
                shutil.rmtree(save_dir, ignore_errors=True)
            else:
                logger.info(f"Directory {save_dir} does not exist. Creating.")
            os.makedirs(save_dir, exist_ok=True)

            for subset in dataset_conf.subsets:
                generations_path = os.path.join(
                    config.base.generations_dir,
                    model_id,
                    dataset_name,
                    prompt_version,
                    f"{subset}_generations.csv",
                )
                logger.info(f"Loading generations from {generations_path}")

                statements = load_statements(generations_path)

                for idx in range(0, len(statements), batch_size):
                    chunk = statements[idx : idx + batch_size]

                    if isinstance(config.capture_activations.input_type, str):
                        config.capture_activations.input_type = [
                            config.capture_activations.input_type
                        ]

                    for input_type in config.capture_activations.input_type:
                        if input_type not in ["prompt_only", "prompt_answer"]:
                            raise ValueError(
                                f"Unknown input_type: {config.capture_activations.input_type}"
                            )

                        batch_inputs = []
                        for statement in [str(s) for s in chunk]:
                            if input_type == "prompt_only":
                                batch_inputs.append(statement[0])
                            elif input_type == "prompt_answer":
                                batch_inputs.append(" ".join(statement))

                        with torch.no_grad():
                            acts = get_acts(
                                batch_inputs,
                                tokenizer,
                                model,
                                layers,
                                device=device,
                            )

                            for l in layers:
                                save_path = os.path.join(
                                    save_dir,
                                    subset,
                                    input_type,
                                    f"layer_{l}",
                                )
                                os.makedirs(save_path, exist_ok=True)
                                torch.save(
                                    acts[l],
                                    os.path.join(
                                        save_path, f"activations_batch_{idx}.pt"
                                    ),
                                )

                            logger.info(
                                f"Saved activations for subset `{subset}`, batch `{idx}`"
                            )

                            logger.info("Emptying CUDA cache")
                            torch.cuda.empty_cache()


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument("--model", dest="model", required=True)
    args_parser.add_argument(
        "--layers", dest="layers", nargs="+", default=None, type=int
    )
    args_parser.add_argument(
        "--batch-size", dest="batch_size", default=1, type=int
    )
    args = args_parser.parse_args()
    capture_activations(args.config, args.model, args.layers, args.batch_size)
