from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch as pt


def get_all_experiment_activations_configs_df(base_path):
    all_activation_exp_configs = defaultdict(list)
    for model_path in Path(f"{base_path}/activations").iterdir():
        for dataset_path in model_path.iterdir():
            for prompt_path in dataset_path.iterdir():
                for subset_path in prompt_path.iterdir():
                    for input_type_path in subset_path.iterdir():
                        for layer_path in input_type_path.iterdir():
                            all_activation_exp_configs["model_id"].append(
                                model_path.name
                            )
                            all_activation_exp_configs["dataset_id"].append(
                                dataset_path.name
                            )
                            all_activation_exp_configs["prompt_id"].append(
                                prompt_path.name
                            )
                            all_activation_exp_configs["subset_id"].append(
                                subset_path.name
                            )
                            all_activation_exp_configs["input_type"].append(
                                input_type_path.name
                            )
                            all_activation_exp_configs["layer"].append(
                                int(layer_path.name.split("_")[-1])
                            )
                            all_activation_exp_configs["path"].append(
                                layer_path
                            )
    return pd.DataFrame(all_activation_exp_configs)


def get_experiment_activations_configs_df_subset(base_path, **kwargs):
    activation_exp_configs_df = get_all_experiment_activations_configs_df(
        base_path
    )

    for col, value in kwargs.items():
        if value is None:
            continue
        activation_exp_configs_df = activation_exp_configs_df[
            activation_exp_configs_df[col] == value
        ]
    return activation_exp_configs_df


def load_activations(
    base_path,
    model_id,
    dataset_id,
    prompt_id,
    subset_id,
    input_type,
    layer,
    batch_ids=None,
):
    if batch_ids:
        batch_ids = [int(batch_id) for batch_id in batch_ids]

    paths = sorted(
        list(
            Path(
                f"{base_path}/activations/{model_id}/{dataset_id}/{prompt_id}/{subset_id}/{input_type}/layer_{layer}"
            ).iterdir()
        ),
        key=lambda p: int(p.stem.split("_")[-1]),
    )

    activations_list, indices = [], []
    batch_size = None
    for batch_file in paths:
        batch_id = int(batch_file.stem.split("_")[-1])
        if batch_ids and batch_id not in batch_ids:
            continue

        activations = pt.load(batch_file, map_location=pt.device("cpu"))
        activations_list.append(activations)

        batch_size = activations.shape[0]

        if batch_size is None:
            batch_size = activations.shape[0]
        else:
            assert batch_size == activations.shape[0]

        indices.append(pd.Series(range(batch_size), name="index") + batch_id)

    return (
        pt.cat(activations_list, dim=0),
        pd.concat(indices, axis=0).reset_index(drop=True),
    )


def load_labels(
    base_path, model_id, dataset_id, prompt_id, subset_id, indices=None
):
    paths = list(
        Path(
            f"{base_path}/evaluations/{model_id}/{dataset_id}/{prompt_id}/"
        ).iterdir()
    )
    for path in paths:
        filename = path.stem
        if subset_id != filename.split("_generations_evaluated")[0]:
            continue
        df = pd.read_csv(path)
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)
        return df
    raise ValueError(
        f"No labels found for {model_id} {dataset_id} {prompt_id} {subset_id}"
    )
