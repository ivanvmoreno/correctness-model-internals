from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from classifying import ActivationsHandler, BinaryClassifier, DirectionCalculator


def visualise_clusters_dimenstionality_reduction(
    activations_handler_train: ActivationsHandler,
    activations_handler_test: ActivationsHandler,
    direction_calculator: None | DirectionCalculator = None,
    experiment_path: None | Path | str = None,
):
    pca_pipeline = make_pipeline(StandardScaler(), PCA(n_components=2))
    pca_pipeline.fit(activations_handler_train.activations)
    pca_activations_test = pca_pipeline.transform(activations_handler_test.activations)

    ax = sns.scatterplot(
        x=pca_activations_test[:, 0],
        y=pca_activations_test[:, 1],
        hue=activations_handler_test.labels,
    )

    if direction_calculator is not None:
        correctness_end_pca = pca_pipeline.transform(
            direction_calculator.classifying_direction[None, :]
        ).squeeze()

        zero_pca = pca_pipeline.transform(
            [[0] * activations_handler_train.activations.shape[1]]
        ).squeeze()

        correctness_direction_pca = (
            correctness_end_pca - zero_pca
        )  # need to transform the begining and end of the vector so that we transform the vector so that we can get the difference in pca space.
        ax.quiver(
            0,
            0,
            correctness_direction_pca[0],
            correctness_direction_pca[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="red",
            label="Correctness Direction",
        )

    ax.legend()
    plt.tight_layout()
    if experiment_path:
        plt.savefig(experiment_path / "pca.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.clf()


def visualise_score(
    binary_classifier: BinaryClassifier,
    experiment_path: None | Path = None,
    extra_groups: None | dict[str, Any] = None,
):
    df = pd.DataFrame(
        {
            "label": binary_classifier.test_labels,
            "score": binary_classifier.test_classification_score,
        }
    )
    if extra_groups:
        if any(
            extra_group in df["label"].unique() for extra_group in extra_groups.keys()
        ):
            raise ValueError(
                "extra_groups can't have names of labels that already exist"
            )
        df = pd.concat(
            [df]
            + [
                pd.DataFrame({"label": group_name, "vscorealue": group})
                for group_name, group in extra_groups.items()
            ],
            axis=0,
        )

    optimal_cut = binary_classifier.optimal_cut
    acc = binary_classifier.classification_metrics["accuracy_score"]
    f1 = binary_classifier.classification_metrics["f1_score"]

    # Create the plot with multiple categories but shared bins
    bin_edges = np.histogram_bin_edges(df["score"], bins=100)
    ax = sns.histplot(
        data=df,
        x=df["score"],
        bins=bin_edges,
        hue=df["label"],
    )

    # Add the threshold line
    optimal_cut_label = f"Optimal Threshold {optimal_cut:.3f}"
    ax.axvline(optimal_cut, color="red", linestyle="--", label=optimal_cut_label)

    # Create the new legend with all elements
    ax.legend()
    ax.set_title(f"Classification. Accuracy={acc:.3f}, F1={f1:.3f}")
    plt.tight_layout()
    if experiment_path:
        plt.savefig(
            experiment_path / "classifier_separation.png", dpi=300, bbox_inches="tight"
        )
    else:
        plt.show()
    plt.clf()


def visualise_roc(binary_classifier, experiment_path: None | Path = None):
    plt.figure(figsize=(8, 6))
    plt.plot(
        binary_classifier.test_false_positive_rate,
        binary_classifier.test_true_positive_rate,
        label=f"ROC curve (AUC = {binary_classifier.roc_auc:.4f})",
    )

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if experiment_path:
        plt.savefig(
            experiment_path / "classifier_roc.png", dpi=300, bbox_inches="tight"
        )
    else:
        plt.show()
    plt.clf()
