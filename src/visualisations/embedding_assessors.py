import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)

DATASET_COLORS = {
    "trivia_qa_2_60k": "#000000",
    "cities_10k": "#e41a1c",
    "gsm8k": "#4daf4a",
    "math_operations_6k": "#377eb8",
    "medals_9k": "#ff7f00",
    "birth_years_4k": "#984ea3",
}
DATASET_MARKERS = {
    "trivia_qa_2_60k": "o",
    "cities_10k": "s",
    "gsm8k": "D",
    "math_operations_6k": "^",
    "medals_9k": "X",
    "birth_years_4k": "P",
}
DATASET_NAME_MAP = {
    "trivia_qa_2_60k": "TriviaQA 2 60k",
    "cities_10k": "Cities 10k",
    "gsm8k": "GSM8k",
    "math_operations_6k": "Math Ops 6k",
    "medals_9k": "Medals 9k",
    "birth_years_4k": "Birth Years 4k",
}
MODEL_NAME_MAP = {
    "llama3.1_8b_chat": "Llama 3.1 8B",
    "llama3.3_70b": "Llama 3.3 70B",
    "mistral_7b_instruct": "Mistral 7B",
    "ministral_8b_instruct": "Ministral 8B",
    "qwen_2.5_7b_instruct": "Qwen 2.5 7B",
    "deepseek_qwen_32b": "Qwen 32B (DeepSeek)",
}


def load_metrics_json(
    metrics_path: str | Path,
    which: str = "aggregated",
    metric_name: str = "auc_roc",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    metrics_path : str | Path
    which : {"aggregated", "by_dataset"}
        Whether to take the global test-set metric or one line per individual
        test dataset.
    metric_name : str
        Key inside the `metrics` dict to extract (e.g. "auc_roc", "accuracy").

    Returns
    -------
    pd.DataFrame with columns
        ["model_name", "train_dataset", "test_dataset", "metric"]
    """
    p = Path(metrics_path)
    if not p.exists():
        raise FileNotFoundError(p)

    with p.open() as f:
        d = json.load(f)

    model_name = d["config"]["model_id"]
    train_datasets = d["config"]["train_datasets"]
    # there is always exactly one train dataset in the current workflow
    train_dataset = train_datasets[0]

    rows = []

    if which == "aggregated":
        mval = d["evaluation_metrics"]["aggregated"][metric_name]
        rows.append(
            dict(
                model_name=model_name,
                train_dataset=train_dataset,
                test_dataset="aggregated",
                metric=mval,
                top_k=d["config"]["top_k"],
            )
        )
    elif which == "by_dataset":
        for test_dataset, info in d["evaluation_metrics"]["by_dataset"].items():
            mval = info["metrics"][metric_name]
            rows.append(
                dict(
                    model_name=model_name,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    metric=mval,
                    top_k=d["config"]["top_k"],
                )
            )
        if "in_distribution" in d["evaluation_metrics"]:
            mval = d["evaluation_metrics"]["in_distribution"]["metrics"][
                metric_name
            ]
            rows.append(
                dict(
                    model_name=model_name,
                    train_dataset=train_dataset,
                    test_dataset=train_dataset,
                    metric=mval,
                    top_k=d["config"]["top_k"],
                )
            )
    else:
        raise ValueError("`which` must be 'aggregated' or 'by_dataset'.")

    return pd.DataFrame(rows)


def plot_assessor_metrics(
    metrics_json_paths: dict[str, str | Path],
    metric_name: str = "auc_roc",
    which: str = "by_dataset",
    figsize: tuple[int, int] = (7, 10),
    save_path: str | None = None,
) -> None:
    """
    Parameters
    ----------
    metrics_json_paths : dict
        Mapping *label → path_to_metrics.json*.
        The *label* (left side) is what will appear on the x-axis.
        E.g. {"Llama-3 8B (top-1024)": "…/llama3_8b_chat_birth_years…metrics.json"}
    metric_name : str
        Metric key stored in each json ("auc_roc", "accuracy", …).
    which : {"aggregated", "by_dataset"}
        Same meaning as in `load_metrics_json`.
    """
    dfs = []
    for label, path in metrics_json_paths.items():
        df = load_metrics_json(path, which=which, metric_name=metric_name)
        df["label"] = label
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    agg_dfs = []
    for label, path in metrics_json_paths.items():
        df_agg = load_metrics_json(
            path, which="aggregated", metric_name=metric_name
        )
        df_agg["label"] = label
        agg_dfs.append(df_agg)
    df_agg_all = pd.concat(agg_dfs, ignore_index=True)

    label_order = list(metrics_json_paths.keys())
    df_all["label"] = pd.Categorical(
        df_all["label"], categories=label_order, ordered=True
    )

    n_rows = df_all["train_dataset"].nunique()
    n_labels = len(label_order)
    if figsize is None:
        per_label_w = 1.2  # inches per item on the x-axis
        min_width = 6
        width = max(min_width, n_labels * per_label_w)
        height = 3 * n_rows  # ~3 inches per subplot row
        figsize = (width, height)

    base_height = n_rows * 5
    base_width = base_height * 1.3
    fig, axes = plt.subplots(
        n_rows,
        1,
        sharex=True,
        figsize=(base_width, base_height),
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (train_ds, grp) in zip(axes, df_all.groupby("train_dataset")):
        for test_ds, sub in grp.groupby("test_dataset"):
            sns.scatterplot(
                data=sub,
                x="label",
                y="metric",
                ax=ax,
                marker=DATASET_MARKERS.get(test_ds, "o"),
                color=DATASET_COLORS.get(test_ds, "#333333"),
                s=50,
                label=DATASET_NAME_MAP.get(test_ds, test_ds),
            )

        subset_agg = df_agg_all[df_agg_all["train_dataset"] == train_ds]
        sns.scatterplot(
            data=subset_agg,
            x="label",
            y="metric",
            ax=ax,
            marker="*",
            color="yellow",
            s=150,
            label="Aggregated",
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_ylabel(
            f"Test {metric_name.replace('_', ' ').upper()}"
            f"\n(Trained on {DATASET_NAME_MAP[train_ds]})"
        )
        y_min, y_max = grp["metric"].min(), grp["metric"].max()
        center = 0.5 * (y_min + y_max)
        lower = max(0.0, center - 0.25)
        upper = min(1.0, center + 0.25)
        ax.set_ylim(lower, upper)
        ax.axhline(
            0.5, color="gray", linestyle="--", linewidth=1, label="Random Guess"
        )
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_facecolor("white")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Dataset / Summary",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
    )
    for ax in axes:
        ax.get_legend().remove()

    axes[-1].set_xlabel("Model")
    fig.suptitle(
        f"Embedding Assessor Performance – Top-K: {df_all['top_k'].unique()[0]} Dimensions",
    )

    strip_suffix = re.compile(r"\s*\(.*\)$")
    for ax in axes:
        new_labels = []
        for tick in ax.get_xticklabels():
            raw = strip_suffix.sub("", tick.get_text())
            pretty = MODEL_NAME_MAP.get(raw, raw)
            new_labels.append(pretty)
        ax.set_xticklabels(new_labels, rotation=0)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
