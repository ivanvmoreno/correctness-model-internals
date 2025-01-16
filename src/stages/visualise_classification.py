import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch as pt
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _act_file_to_batch_idx(file: Path) -> int:
    return int(str(file).split("_")[-1].split(".")[0])


def calculate_correctness_direction(activations, label_df):
    # activation = mu + sign * correctness_direction where mu is the mean of all activations (the centroid) and sign is -1 if incorrect and 1 if correct
    # Basically the centroid over the data + the correctness direction (or it flipped) should take you to the centroid of the class.
    mu = activations.mean(axis=0)

    s = pt.ones(activations.shape[0])
    s[~label_df["correct"]] = (
        -1
    )  # those that are incorrect should point in the opposite direction

    correctness_direction = pt.mean((activations - mu) * s[:, None], dim=0)
    return (
        correctness_direction,
        (activations - mu) @ correctness_direction,
    )  # From algebra should be /s, but x/1=x*1 and x/-1=x*-1 so can just multiply by s


def evaluate_classification(labels, correctness, experiment_path, extra_info=None):
    df = pd.DataFrame({"label": labels, "correctness": correctness})

    fpr, tpr, thresholds = roc_curve(df["label"], df["correctness"])
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]

    acc = accuracy_score(df["label"], df["correctness"] > optimal_threshold)
    roc_auc = auc(fpr, tpr)

    ax = sns.histplot(df[~df["label"]]["correctness"], label="Incorrect")
    ax = sns.histplot(df[df["label"]]["correctness"], label="Correct", ax=ax)
    ax.axvline(
        optimal_threshold,
        color="red",
        linestyle="--",
        label=f"Optimal Threshold {optimal_threshold:.3f}, Accuracy {acc:.3f}",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        experiment_path / "classifier_separation.png", dpi=300, bbox_inches="tight"
    )

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.scatter(
        fpr[optimal_idx],
        tpr[optimal_idx],
        color="red",
        label=f"Optimal Threshold = {optimal_threshold:.2f}",
    )
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(experiment_path / "classifier_roc.png", dpi=300, bbox_inches="tight")

    experiment_path.write

    logging_config = {
        **{
            "accuracy": acc,
            "roc_auc": roc_auc,
            "optimal_threshold": optimal_threshold,
        },
        **(extra_info or {}),
    }
    with experiment_path.open("w", encoding="utf-8") as file:
        json.dump(logging_config, file, indent=4, ensure_ascii=False)


def load_labels_df_and_activations(
    layer: str,
    activations_dir: str,
    question_answer_file: str,
    first_batch: str | None = None,
    n_batches: str | None = None,
    batch_size: int = 25,
):
    labels_df = pd.read_csv(question_answer_file)

    if first_batch is not None:
        if first_batch % batch_size != 0:
            raise ValueError("first_batch must be a multiple of batch_size")

        labels_df = labels_df.iloc[
            first_batch
            * batch_size : ((first_batch + n_batches) * batch_size if n_batches else -1)
        ]
    elif n_batches is not None:
        raise TypeError("Only provide n_batches if first_batch is provided")

    # todo assumes that the last number in the file name is the index of the first entry in the batch, and that they're batched in 25 entries
    tensors = []
    for file in sorted(
        Path(activations_dir).iterdir(),
        key=lambda file_: _act_file_to_batch_idx(file_),
    ):
        if f"layer_{layer}" not in str(file):
            continue

        batch_first_entry_index = _act_file_to_batch_idx(file)
        if first_batch is not None:
            if first_batch > batch_first_entry_index:
                continue
            if (
                n_batches is not None
                and batch_first_entry_index > first_batch + n_batches
            ):
                continue

        act_tensor = pt.load(file)
        tensors.append(act_tensor)

    activations = pt.cat(tensors, axis=0).to("cpu")

    if len(labels_df) != activations.shape[0]:
        raise RuntimeError(
            f"Number of labels ({len(labels_df)}) does not match number of activations ({activations.shape[0]})"
        )

    return labels_df, activations


def evaluate_correctness_direction_and_classifier(
    activations, labels_df, experiment_path
):
    pca_pipeline = make_pipeline(StandardScaler(), PCA(n_components=2))
    pca_activations = pca_pipeline.fit_transform(activations)

    correctness_direction, labels_df["correctness"] = calculate_correctness_direction(
        activations, labels_df
    )

    correctness_end_pca = pca_pipeline.transform(
        correctness_direction[None, :]
    ).squeeze()

    zero_pca = pca_pipeline.transform([[0] * activations.shape[-1]]).squeeze()
    correctness_direction_pca = (
        correctness_end_pca - zero_pca
    )  # need to transform the begining and end of the vector so that we transform the vector so that we can get the difference in pca space.

    ax = sns.scatterplot(
        x=pca_activations[:, 0], y=pca_activations[:, 1], hue=labels_df["correct"]
    )
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
    plt.savefig(experiment_path / "pca.png", dpi=300, bbox_inches="tight")

    evaluate_classification(
        labels_df["correct"],
        labels_df["correctness"],
        experiment_path / "correctness_direction",
    )


def train_and_evaluate_mlp_classifier(
    labels_df,
    activations,
    experiment_path,
    verbose=False,
    train_frac=0.8,
    n_epochs=100,
    lr=0.01,
):
    train_indices = np.random.uniform(0, 1, len(labels_df)) < train_frac
    test_indices = ~train_indices

    train_label_df = labels_df.iloc[train_indices]
    test_label_df = labels_df.iloc[test_indices]

    class SingleLayerPerceptron(nn.Module):
        def __init__(self, input_size, output_size):
            super(SingleLayerPerceptron, self).__init__()
            self.fc = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, output_size)

        def forward(self, x):
            return self.fc2(nn.functional.relu(self.fc(x)))

    model = SingleLayerPerceptron(activations.shape[-1], 1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        outputs = model(activations[train_indices])

        # Convert targets to float tensor
        targets = pt.tensor(train_label_df["correct"].to_numpy(), dtype=pt.float32)

        # Ensure outputs match target shape
        outputs = outputs.squeeze()

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if verbose and epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

    test_label_df["mlp_classifier_pred"] = (
        pt.nn.functional.sigmoid(model(activations[test_indices]))
        .detach()
        .numpy()
        .squeeze()
    )

    evaluate_classification(
        test_label_df["correct"],
        test_label_df["mlp_classifier_pred"],
        experiment_path=experiment_path / "mlp_classifier",
        extra_info={
            "model_architecture": str(model),
            "train_frac": train_frac,
            "n_epochs": n_epochs,
            "lr": lr,
        },  # Note that this isn't intended for logging variation to this classifier, just for logging the parameters used.
    )


def get_experiment_path(output_path, _experiment_id=None):
    if _experiment_id is not None:
        print(
            f"Warning, setting {_experiment_id=}, if this already exists it will be overwritten"
        )
        return Path(output_path) / str(_experiment_id)

    output_path = Path(output_path)
    max_experiment = 0
    for experiment_dir in output_path.iterdir():
        try:
            max_experiment = max(max_experiment, int(experiment_dir.name))
        except Exception:
            continue
    experiment_path = output_path / str(max_experiment + 1)
    if experiment_path.exists():
        raise RuntimeError(f"Experiment path {experiment_path} already exists")

    experiment_path.mkdir()
    return experiment_path


def classifier_experiment_run(
    model: str,
    layer: str,
    activations_dir: str,
    question_answer_file: str,
    first_batch: str | None = None,
    n_batches: str | None = None,
    batch_size: int = 25,
    check_correctness_classifier: bool = True,
    check_mlp_classifier: bool = True,
    output_path: str = "./experiment_results",
    notes: str = "",
    _experiment_id: int = None,  # careful, only use this to overwrite existing experiments
) -> None:
    # todo: update to save plots and stats to file.

    labels_df, activations = load_labels_df_and_activations(
        layer=layer,
        activations_dir=activations_dir,
        question_answer_file=question_answer_file,
        first_batch=first_batch,
        n_batches=n_batches,
        batch_size=batch_size,
    )
    if not check_correctness_classifier and not check_mlp_classifier:
        raise ValueError(
            "At least one of check_correctness_classifier or check_mlp_classifier must be True"
        )

    experiment_path = get_experiment_path(output_path, _experiment_id=_experiment_id)

    logging_config = {
        "model": model,
        "layer": layer,
        "activations_dir": activations_dir,
        "question_answer_file": question_answer_file,
        "first_batch": first_batch,
        "n_batches": n_batches,
        "batch_size": batch_size,
        "check_correctness_classifier": check_correctness_classifier,
        "check_mlp_classifier": check_mlp_classifier,
        "timestamp": str(datetime.now()),
        "notes": notes,
    }
    with experiment_path.open("w", encoding="utf-8") as file:
        json.dump(logging_config, file, indent=4, ensure_ascii=False)

    if check_correctness_classifier:
        evaluate_correctness_direction_and_classifier(
            activations, labels_df, experiment_path
        )

    if check_mlp_classifier:
        train_and_evaluate_mlp_classifier(labels_df, activations, experiment_path)


if __name__ == "__main__":
    print("Run this from the command line at the base level of the repo")
    args_parser = argparse.ArgumentParser()
    # args_parser.add_argument("--config", dest="config", required=True) # is this needed?
    args_parser.add_argument("--model", dest="model", required=True)
    args_parser.add_argument("--layer", dest="layer", required=True)
    args_parser.add_argument("--activations_dir", dest="head", required=True)
    args_parser.add_argument(
        "--question-answer-file", dest="question_answer_file", required=True
    )
    args_parser.add_argument("--first-batch", dest="first_batch", default=None)
    args_parser.add_argument("--n-batches", dest="n_batches", default=None)
    args_parser.add_argument("--notes", dest="notes", default=None)
    args = args_parser.parse_args()
    classifier_experiment_run(
        model=args.model,
        activations_dir=args.activations_dir,
        question_answer_file=args.question_answer_file,
        first_batch=args.first_batch,
        n_batches=args.n_batches,
        notes=args.notes,
    )
