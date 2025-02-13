from pathlib import Path

import pandas as pd
import torch as pt


def get_all_experiment_activations_configs_df(
    base_path: str | Path,
) -> pd.DataFrame:
    """
    Get a dataframe of all the experiment activations configs.

    Parameters
    ----------
    base_path : str | Path
        The base path to the directory containing the activations dir

    Returns
    -------
    pd.DataFrame
        A dataframe of all the experiment activations configs
    """
    records = []
    base = Path(base_path)

    for path in base.glob("activations/*/*/*/*/*/*"):  # 6 directory levels deep
        parts = path.parts
        records.append(
            {
                "model_id": parts[-6],  # activations/[model]
                "dataset_id": parts[-5],  # activations/[model]/[dataset]
                "prompt_id": parts[-4],  # .../[prompt]
                "subset_id": parts[-3],  # .../[subset]
                "input_type": parts[-2],  # .../[input_type]
                "layer": int(parts[-1].split("_")[-1]),
                "path": path,
            }
        )

    return pd.DataFrame(records)


def get_experiment_activations_configs_df_subset(
    base_path: str | Path, **kwargs
) -> pd.DataFrame:
    """
    Get a dataframe of all the experiment activations configs that match the given
    criteria.

    Parameters
    ----------
    base_path : str | Path
        The base path to the directory containing the activations dir
    **kwargs : dict
        The criteria to filter the dataframe by
        eg. model_id="llama3" etc. where the arg name is the column name and the value
        is the value to filter by.

    Returns
    -------
    pd.DataFrame
        A dataframe of all the experiment activations configs that match the given
        criteria
    """
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
    base_path: str | Path,
    model_id: str,
    dataset_id: str,
    prompt_id: str,
    subset_id: str,
    input_type: str,
    layer: int,
    batch_ids: list[int] | None = None,
) -> tuple[pt.Tensor, pd.Series]:
    """
    Load the activations for a given run

    Parameters
    ----------
    base_path : str | Path
        The base path to the directory containing the activations dir
    model_id : str
        The model id
    dataset_id : str
        The dataset id
    prompt_id : str
        The prompt id
    subset_id : str
        The subset id
    input_type : str
        The input type
    layer : int
        The layer
    batch_ids : list[int] | None
        The batch ids to load
        eg. [1, 2, 3] will only load the activations for batch ids 1, 2, and 3
        If None, all batch ids will be loaded

    Returns
    -------
    tuple[pt.Tensor, pd.Series]
        A tuple of:
        - the activations
        - the indices (useful if you don't load all activations)
    """
    if batch_ids:
        batch_ids = [int(batch_id) for batch_id in batch_ids]

    paths = sorted(
        list(
            Path(
                f"{base_path}/activations/{model_id}/{dataset_id}/{prompt_id}"
                f"/{subset_id}/{input_type}/layer_{layer}"
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
    base_path: str | Path,
    model_id: str,
    dataset_id: str,
    prompt_id: str,
    subset_id: str,
    indices: list[int] | None = None,
) -> pd.DataFrame:
    """
    Load the labels for a given run

    Parameters
    ----------
    base_path : str | Path
        The base path to the directory containing the evaluations dir
    model_id : str
        The model id
    dataset_id : str
        The dataset id
    prompt_id : str
        The prompt id
    subset_id : str
        The subset id
    indices : list[int] | None
        The indices to load
        eg. [1, 2, 3] will only load the labels for batch ids 1, 2, and 3
        If None, all labels will be loaded

    Returns
    -------
    pd.DataFrame
        A dataframe of the labels
    """
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
