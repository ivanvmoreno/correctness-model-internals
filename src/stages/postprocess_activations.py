import argparse
import os
import shutil

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
    logger = get_logger("GENERATE_ACTS", config.base.log_level)

    logger.info(f"Postprocessing activations for model {model_id}")

    if layers is None:
        layers = list(range(len(model.model.layers)))

    if isinstance(layers, int):
        layers = [layers]
    layers = [int(layer) for layer in layers]

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
                            config.base.generations_dir,
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

                        num_files = len(os.listdir(activations_path))
                        activations_list = []
                        for i in range(num_files):
                            activations_batch_path = os.path.join(
                                activations_path,
                                f"activations_batch_{i}.pt",
                            )
                            activations_batch = load_activations(
                                activations_batch_path
                            ).numpy()
                            activations_list.append(activations_batch)
                        activations_joined = np.concatenate(
                            activations_list, axis=0
                        )

                        logger.info(
                            f"Saving joined activations to {activations_path}"
                        )
                        np.save(
                            os.path.join(
                                activations_path,
                                f"activations_joined.npy",
                            ),
                            activations_joined,
                        )

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
                                f"activations_{method}.pt",
                            )
                            logger.info(
                                f"Saving postprocessed activations to {save_path}"
                            )
                            np.save(save_path, processed_activations)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument("--model", dest="model", required=True)
    args_parser.add_argument(
        "--layers", dest="layers", nargs="+", default=None, type=int
    )
    args = args_parser.parse_args()
    postprocess_activations(
        args.config, args.model, args.layers, args.batch_size
    )
