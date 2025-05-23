import argparse
import json
import os
import shutil

import torch

from src.data import load_statements
from src.model import get_acts, load_hf_model
from src.utils.config import load_config
from src.utils.logging import get_logger


def capture_activations(
    config_path: str,
    model_id: str,
    layers: list,
    batch_size: int = 25,
    device: str = "cuda",
) -> None:
    """Format datasets for question-answering tasks by capturing model activations.

    Args:
        config_path (str): Path to configuration file
        model_id (str): Model to use for generation
        layers (list): List of layers to extract activations from
        batch_size (int, optional): Batch size for generation. Defaults to 25.
        device (str, optional): Device to run inference on. Defaults to 'cuda'.
    """
    config = load_config(config_path)
    model_config = config.models[model_id]
    logger = get_logger("CAPTURE_ACTS", config.base.log_level)

    logger.info(f"Generating activations for model {model_id}")

    logger.info(f"Loading model into GPU (device={device})")
    tokenizer, model = load_hf_model(
        config.base.models_dir, model_config.dir_path
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # IMPORTANT: right padded, as we're capturing the activations for generated sequences

    if layers is None:
        raise ValueError(
            "Please provide a list of layers to extract activations."
        )

    if isinstance(layers, str):
        if layers == "-1":
            layers = list(range(model_config.num_layers))
        else:
            try:
                layers = json.loads(layers)
            except json.JSONDecodeError:
                layers = [layers]
            layers = [int(layer) for layer in layers]

    # Skip layers based on step size
    layers = layers[:: config.capture_activations.step_size]

    logger.info(f"Extracting activations from layers {layers}")

    for dataset_name, dataset_conf in config.datasets.items():
        for prompt_version, _ in dataset_conf.prompts.items():
            save_dir = os.path.join(
                config.base.activations_dir,
                config.capture_activations.raw_dir_path,
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

                for batch_num, idx in enumerate(
                    range(0, len(statements), batch_size)
                ):
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
                        for s in chunk:
                            if input_type == "prompt_only":
                                batch_inputs.append(s[0])
                            elif input_type == "prompt_answer":
                                batch_inputs.append(
                                    " ".join([str(x) for x in s])
                                )

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
                                f"Saved activations for dataset {dataset_name}, prompt_version {prompt_version}, subset {subset}, input_type {input_type}, batch {batch_num}"
                            )

                            torch.cuda.empty_cache()


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument(
        "--model",
        dest="model",
        required=True,
        nargs="+",  # Allow multiple models
        help="Model ID(s) to use for capturing activations. Can be single or multiple models.",
    )
    args_parser.add_argument("--layers", dest="layers", default="-1", type=str)
    args_parser.add_argument(
        "--batch-size", dest="batch_size", default=5, type=int
    )
    args = args_parser.parse_args()

    # Handle both single model and multiple models
    models = args.model if isinstance(args.model, list) else [args.model]

    # Run activation capture for each model
    for model_id in models:
        capture_activations(args.config, model_id, args.layers, args.batch_size)
