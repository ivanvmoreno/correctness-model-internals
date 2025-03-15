import argparse
import json
import os
import shutil
from typing import Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.data import load_activations
from src.utils.config import load_config
from src.utils.logging import get_logger


def postprocess_activations(
    config_path: str,
    model_id: str,
    layers: list,
) -> None:
    """Postprocess activations (PCA, t-SNE) from generated statements.

    Args:
        config_path (str): Path to configuration file
        model_id (str): Model to use for generation
        layers (list): List of layers to extract activations from
    """
    config = load_config(config_path)
    model_config = config.models[model_id]
    logger = get_logger("POSTPROCESS_ACTS", config.base.log_level)

    logger.info(f"Postprocessing activations for model {model_id}")

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

    logger.info(f"Postprocessing activations from layers {layers}")

    for dataset_name, dataset_conf in config.datasets.items():
        for prompt_version, _ in dataset_conf.prompts.items():
            for subset in dataset_conf.subsets:
                for layer in layers:
                    if isinstance(config.capture_activations.input_type, str):
                        config.capture_activations.input_type = [
                            config.capture_activations.input_type
                        ]

                    for input_type in config.capture_activations.input_type:
                        save_dir = os.path.join(
                            config.base.activations_dir,
                            "postprocessed",
                            model_id,
                            dataset_name,
                            prompt_version,
                            subset,
                            input_type,
                            f"layer_{layer}",
                        )
                        if os.path.exists(save_dir):
                            logger.info(
                                f"Directory {save_dir} exists. Clearing previous runs."
                            )
                            shutil.rmtree(save_dir, ignore_errors=True)
                        else:
                            logger.info(
                                f"Directory {save_dir} does not exist. Creating."
                            )
                        os.makedirs(save_dir, exist_ok=True)

                        activations_path = os.path.join(
                            config.base.activations_dir,
                            "raw",
                            model_id,
                            dataset_name,
                            prompt_version,
                            subset,
                            input_type,
                            f"layer_{layer}",
                        )
                        logger.info(
                            f"Loading activations from {activations_path}"
                        )

                        batch_files = os.listdir(activations_path)
                        activations_list = []
                        for batch_fname in batch_files:
                            activations_batch_path = os.path.join(
                                activations_path,
                                batch_fname,
                            )
                            activations_batch = load_activations(
                                activations_batch_path,
                            )
                            activations_list.append(activations_batch)
                        activations_joined = np.concatenate(
                            activations_list, axis=0
                        )

                        activations_joined_path = os.path.join(
                            save_dir,
                            "activations_joined.npy",
                        )
                        logger.info(
                            f"Saving joined activations to {activations_joined_path}"
                        )
                        np.save(activations_joined_path, activations_joined)

                        # Postprocess activations
                        for method in config.postprocess_activations.methods:
                            logger.info(
                                f"Postprocessing activations using {method}"
                            )
                            method_conf = config.postprocess_activations[method]
                            if method == "pca":
                                pca = PCA(
                                    **method_conf,
                                )
                                processed_activations = pca.fit_transform(
                                    activations_joined
                                )
                            elif method == "tsne":
                                tsne = TSNE(
                                    **method_conf,
                                )
                                processed_activations = tsne.fit_transform(
                                    activations_joined
                                )
                            else:
                                raise ValueError(
                                    f"Unknown postprocessing method {method}"
                                )

                            # Save activations
                            save_path = os.path.join(
                                save_dir,
                                f"activations_{method}.npy",
                            )
                            logger.info(
                                f"Saving postprocessed activations to {save_path}"
                            )
                            np.save(save_path, processed_activations)

                            # Remove joined activations
                            logger.info(
                                f"Removing joined activations from {activations_joined_path}"
                            )
                            os.remove(activations_joined_path)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument(
        "--model",
        dest="model",
        required=True,
        nargs="+",  # Allow multiple models
        help="Model ID(s) to use for postprocessing. Can be single or multiple models.",
    )
    args_parser.add_argument("--layers", dest="layers", default=None, type=str)
    args = args_parser.parse_args()

    # Handle both single model and multiple models
    models = args.model if isinstance(args.model, list) else [args.model]

    # Run postprocessing for each model
    for model_id in models:
        postprocess_activations(args.config, model_id, args.layers)
