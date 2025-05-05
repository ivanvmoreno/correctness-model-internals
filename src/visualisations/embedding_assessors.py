from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, List, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

DATASET_COLOURS: Mapping[str, str] = {
    "trivia_qa_2_60k": "#000000",
    "cities_10k": "#e41a1c",
    "gsm8k": "#4daf4a",
    "math_operations_6k": "#377eb8",
    "medals_9k": "#ff7f00",
    "birth_years_4k": "#984ea3",
}
DATASET_MARKERS: Mapping[str, str] = {
    "trivia_qa_2_60k": "o",
    "cities_10k": "s",
    "gsm8k": "D",
    "math_operations_6k": "^",
    "medals_9k": "X",
    "birth_years_4k": "P",
}
DATASET_DISPLAY: Mapping[str, str] = {
    "trivia_qa_2_60k": "TriviaQA 2 60k",
    "cities_10k": "Cities 10k",
    "gsm8k": "GSM8k",
    "math_operations_6k": "Math Ops 6k",
    "medals_9k": "Medals 9k",
    "birth_years_4k": "Birth Years 4k",
}
MODEL_DISPLAY: Mapping[str, str] = {
    "llama3.1_8b_chat": "Llama 3.1 8B",
    "llama3.3_70b": "Llama 3.3 70B",
    "mistral_7b_instruct": "Mistral 7B",
    "ministral_8b_instruct": "Ministral 8B",
    "qwen_2.5_7b_instruct": "Qwen 2.5 7B",
    "deepseek_qwen_32b": "Qwen 32B (DeepSeek)",
}
CLASSIFIER_DISPLAY: Mapping[str, str] = {
    "logistic": "Logistic Regression",
    "xgboost": "XGBoost",
}

__all__ = [
    "load_metrics_json",
    "plot_assessor_metrics",
]


def dataset_colour(ds: str) -> str:
    return DATASET_COLOURS.get(ds, "#666666")


def dataset_marker(ds: str) -> str:
    return DATASET_MARKERS.get(ds, "o")


def _prettify_model(raw: str) -> str:
    return MODEL_DISPLAY.get(raw, raw)


_strip_suffix = re.compile(r"\s*\(.*\)$")


def load_metrics_json(
    metrics_path: str | Path,
    *,
    which: str = "aggregated",
    metric_name: str = "auc_roc",
) -> pd.DataFrame:
    """Parse metrics.json file into DataFrame."""
    path = Path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r") as fh:
        blob: Dict = json.load(fh)

    cfg = blob["config"]
    ev = blob["evaluation_metrics"]

    model_name: str = cfg["model_id"]
    train_dataset: str = cfg["train_datasets"][0]

    rows: List[Dict] = []

    def _add_row(test_ds: str, value: float | None) -> None:
        rows.append(
            {
                "model_name": model_name,
                "train_dataset": train_dataset,
                "test_dataset": test_ds,
                "metric": value,
                "classifier": cfg["classifier_type"],
                "top_k": cfg["top_k"],
            }
        )

    if which == "aggregated":
        _add_row("aggregated", ev.get("aggregated", {}).get(metric_name))

    elif which == "by_dataset":
        # External test datasets
        for ds, info in ev.get("by_dataset", {}).items():
            _add_row(ds, info.get(metric_name))

        # In‑distribution performance
        held = ev.get("held_out", {})
        held_val = None if held.get("status") else held.get(metric_name)
        if held_val is not None:
            _add_row(train_dataset, held_val)
        else:  # Fallback to CV if needed
            cv_dict = ev.get("cross_validation", {})
            if metric_name in cv_dict:
                _add_row(train_dataset, cv_dict[metric_name])
    else:
        raise ValueError("`which` must be 'aggregated' or 'by_dataset'")

    return pd.DataFrame(rows)


def plot_assessor_metrics(
    *,
    metrics_json_paths: Mapping[str, str | Path],
    metric_name: str = "auc_roc",
    which: str = "by_dataset",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter‑plot assessor metrics for multiple checkpoints."""

    # Load & concat metrics
    frames: List[pd.DataFrame] = []
    for label, path in metrics_json_paths.items():
        df = load_metrics_json(path, which=which, metric_name=metric_name)
        df["label"] = label
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)

    # Order of models on x‑axis follows the insertion order given by caller
    label_order: List[str] = list(metrics_json_paths.keys())
    df_all["label_cat"] = pd.Categorical(
        df_all["label"], categories=label_order, ordered=True
    )

    # Figure geometry
    n_rows = df_all["train_dataset"].nunique()
    fig_height = 6 * n_rows
    fig_width = fig_height * 1.4

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # Pre‑compute helper dicts
    label_to_x = {lab: i for i, lab in enumerate(label_order)}
    test_ds_list = sorted(df_all["test_dataset"].unique())
    n_ds = len(test_ds_list)

    # Offset each dataset slightly for visibility (disabled)
    width = 0.0

    for ax, (train_ds, grp) in zip(axes, df_all.groupby("train_dataset")):
        for j, test_ds in enumerate(test_ds_list):
            sub = grp[grp["test_dataset"] == test_ds]
            if sub.empty:
                continue
            offset = (j - (n_ds - 1) / 2) * width
            x_vals = sub["label_cat"].cat.codes + offset
            ax.scatter(
                x_vals,
                sub["metric"],
                marker=dataset_marker(test_ds),
                color=dataset_colour(test_ds),
                s=55,
                label=DATASET_DISPLAY.get(test_ds, test_ds),
            )

        # Chance baseline & cosmetics
        ax.axhline(
            0.5, color="grey", linestyle="--", linewidth=1, label="Random Guess"
        )
        ax.set_ylabel(
            f"Test {metric_name.replace('_', ' ').upper()}\n(Trained on {DATASET_DISPLAY.get(train_ds, train_ds)})"
        )
        ax.grid(True, linestyle="--", alpha=0.3)

        # Dynamic y‑limits – centre ±0.25 but clamp to [0,1]
        y_vals = grp["metric"].dropna()
        if not y_vals.empty:
            center = y_vals.mean()
            ax.set_ylim(max(0, center - 0.35), min(1, center + 0.35))

        # X‑ticks at integer positions
        ax.set_xticks(list(label_to_x.values()))
        pretty_labels = [
            _prettify_model(_strip_suffix.sub("", lbl)) for lbl in label_order
        ]
        ax.set_xticklabels(pretty_labels, rotation=0)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Dataset",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
    )

    # Figure title
    classifier_name = df_all["classifier"].iloc[0]
    clf_pretty = CLASSIFIER_DISPLAY.get(classifier_name, classifier_name)
    fig.suptitle(f"Embedding‑Assessor Performance ({clf_pretty})", fontsize=14)

    if save_path:
        fig.savefig(Path(save_path), bbox_inches="tight", dpi=300)
    return fig
