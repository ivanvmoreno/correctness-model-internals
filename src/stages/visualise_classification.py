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
from sklearn.metrics import accuracy_score, auc, roc_curve, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def _act_file_to_batch_idx(file: Path) -> int:
    return int(str(file).split("_")[-1].split(".")[0])


def calculate_correctness_direction(activations, label_df):
    # activation = mu + sign * correctness_direction where mu is the mean of all activations (the centroid) and sign is -1 if incorrect and 1 if correct
    # Basically the centroid over the data + the correctness direction (or it flipped) should take you to the centroid of the class.
    mu = activations.mean(axis=0)

    label_df = label_df.reset_index(drop=True)

    s = pt.ones(activations.shape[0])
    s[~label_df["correct"]] = (
        -1
    )  # those that are incorrect should point in the opposite direction

    correctness_direction = pt.mean((activations - mu) * s[:, None], dim=0)
    return (
        correctness_direction,
        # (activations - mu) @ correctness_direction,
        mu,
    )  # From algebra should be /s, but x/1=x*1 and x/-1=x*-1 so can just multiply by s


def evaluate_classification(labels, correctness, experiment_path, optimal_threshold, extra_info=None):
    df = pd.DataFrame({"label": labels, "correctness": correctness})

    fpr, tpr, thresholds = roc_curve(df["label"], df["correctness"])

    acc = accuracy_score(df["label"], df["correctness"] >= optimal_threshold)
    f1 = f1_score(df["label"], df["correctness"] >= optimal_threshold)
    roc_auc = auc(fpr, tpr)

    # Create the plot with multiple categories but shared bins
    bin_edges = np.histogram_bin_edges(df["correctness"], bins=100)
    ax = sns.histplot(
        x=df[~df["label"]]["correctness"], 
        label="Incorrect",
        bins=bin_edges
    )
    ax = sns.histplot(
        x=df[df["label"]]["correctness"], 
        label="Correct",
        bins=bin_edges
    )

    # Add the threshold line
    optimal_threshold_label = f"Optimal Threshold {optimal_threshold:.3f}"
    ax.axvline(
        optimal_threshold,
        color="red",
        linestyle="--",
        label=optimal_threshold_label
    )

    # Create the new legend with all elements
    ax.legend()
    ax.set_title(f"Classification. Accuracy={acc:.3f}, F1={f1:.3f}")
    plt.tight_layout()
    plt.savefig(
        experiment_path / "classifier_separation.png", dpi=300, bbox_inches="tight"
    )
    plt.clf()

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")

    # plt.scatter(
    #     fpr[optimal_idx],
    #     tpr[optimal_idx],
    #     color="red",
    #     label=f"Optimal Threshold = {optimal_threshold:.2f}",
    # )
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(experiment_path / "classifier_roc.png", dpi=300, bbox_inches="tight")
    plt.clf()

    logging_config = {
        **{
            "accuracy": float(acc),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "optimal_threshold": float(optimal_threshold),
        },
        **(extra_info or {}),
    }
    with (experiment_path / "info.json").open("w", encoding="utf-8") as file:
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
        labels_df = labels_df.iloc[
            first_batch
            * batch_size : ((first_batch + n_batches) * batch_size if n_batches else -1)
        ]
    elif n_batches is not None:
        raise TypeError("Only provide n_batches if first_batch is provided")

    # todo assumes that the last number in the file name is the index of the first entry in the batch, and that they're batched in 25 entries
    tensors = []
    any_for_layer = False
    for file in sorted(
        Path(activations_dir).iterdir(),
        key=lambda file_: _act_file_to_batch_idx(file_),
    ):
        if f"layer_{layer}" not in str(file):
            continue
        any_for_layer = True

        batch_first_entry_index = _act_file_to_batch_idx(file)
        if first_batch is not None:
            if first_batch * 25 > batch_first_entry_index + 25 - 1:
                continue
            if (
                n_batches is not None
                and batch_first_entry_index > (first_batch + n_batches - 1) * 25
            ):
                continue

        act_tensor = pt.load(file)
        tensors.append(act_tensor)

    if not any_for_layer:
        raise FileNotFoundError(f"No activations found for layer {layer}")
    activations = pt.cat(tensors, axis=0).to("cpu")

    if len(labels_df) != activations.shape[0]:
        raise RuntimeError(
            f"Number of labels ({len(labels_df)}) does not match number of activations ({activations.shape[0]})"
        )

    return labels_df.reset_index(drop=True), activations

def find_optimal_cut(labels, classifications):
    fpr, tpr, thresholds = roc_curve(labels, classifications)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index[1:]) + 1
    return thresholds[optimal_idx]

def evaluate_correctness_direction_and_classifier(
    X_train,
    X_test,
    train_label_df,
    test_label_df,
    experiment_path,
    # train_frac=0.8,
):
    # train_indices = np.random.uniform(0, 1, len(labels_df)) < train_frac
    # test_indices = ~train_indices

    # train_label_df = labels_df.iloc[train_indices]
    # test_label_df = labels_df.iloc[test_indices]

    pca_pipeline = make_pipeline(StandardScaler(), PCA(n_components=2))
    _ = pca_pipeline.fit_transform(X_train)
    pca_activations_test = pca_pipeline.transform(X_test)

    correctness_direction, mu = calculate_correctness_direction(X_train, train_label_df)
    test_label_df["correctness"] = (X_test - mu) @ correctness_direction
    train_label_df["correctness"] = (X_train - mu) @ correctness_direction

    # labels_df["correctness"]

    correctness_end_pca = pca_pipeline.transform(
        correctness_direction[None, :]
    ).squeeze()

    zero_pca = pca_pipeline.transform([[0] * X_train.shape[-1]]).squeeze()
    correctness_direction_pca = (
        correctness_end_pca - zero_pca
    )  # need to transform the begining and end of the vector so that we transform the vector so that we can get the difference in pca space.

    test_label_df_ = test_label_df.copy()
    test_label_df_.loc[test_label_df_["special"], "correct"] = "IDK" ### todo clean this. only for plotting i don't know responses separately, dataset specific
    ax = sns.scatterplot(
        x=pca_activations_test[:, 0],
        y=pca_activations_test[:, 1],
        hue=test_label_df_["correct"].map(lambda val: {True: "Correct", False: "Incorrect"}.get(val, val)),
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
    plt.clf()

    correctness_direction_path = experiment_path / "correctness_direction"
    correctness_direction_path.mkdir(parents=True, exist_ok=True)

    evaluate_classification(
        test_label_df["correct"],
        test_label_df["correctness"],
        correctness_direction_path,
        optimal_threshold=find_optimal_cut(train_label_df["correct"], train_label_df["correctness"]),
    )


def evaluate_logistic_regression_classifier(
    # activations,
    X_train,
    X_test,
    train_label_df,
    test_label_df,
    experiment_path,
    # train_frac=0.8,
):
    # train_indices = np.random.uniform(0, 1, len(labels_df)) < train_frac
    # test_indices = ~train_indices

    # train_label_df = labels_df.iloc[train_indices]
    # test_label_df = labels_df.iloc[test_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    model = LogisticRegression(random_state=42, solver="lbfgs", max_iter=1000)
    model.fit(X_train, train_label_df["correct"])

    # Step 4: Make predictions
    train_label_df["correctness"] = 1-model.predict_proba(X_train)
    test_label_df["correctness"] = 1-model.predict_proba(X_test)

    correctness_direction_path = experiment_path / "logistic_regression_classifier"
    correctness_direction_path.mkdir(parents=True, exist_ok=True)

    evaluate_classification(
        test_label_df["correct"],
        test_label_df["correctness"],
        correctness_direction_path,
        optimal_threshold=find_optimal_cut(train_label_df["correct"], train_label_df["correctness"]),
        extra_info={
            "model": str(model),
        },
    )


# def train_and_evaluate_mlp_classifier(
#     labels_df,
#     activations,
#     experiment_path,
#     verbose=False,
#     train_frac=0.8,
#     n_epochs=100,
#     lr=0.01,
# ):
#     train_indices = np.random.uniform(0, 1, len(labels_df)) < train_frac
#     test_indices = ~train_indices

#     train_label_df = labels_df.iloc[train_indices]
#     test_label_df = labels_df.iloc[test_indices]

#     class SingleLayerPerceptron(nn.Module):
#         def __init__(self, input_size, output_size):
#             super(SingleLayerPerceptron, self).__init__()
#             self.fc = nn.Linear(input_size, 128)
#             self.fc2 = nn.Linear(128, output_size)

#         def forward(self, x):
#             return self.fc2(nn.functional.relu(self.fc(x)))

#     model = SingleLayerPerceptron(activations.shape[-1], 1).to(pt.float32)

#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.SGD(model.parameters(), lr=lr)

#     for epoch in range(n_epochs):
#         optimizer.zero_grad()
#         # print(f"{model.dtype=}")
#         outputs = model(activations[train_indices].to(pt.float32))

#         # Convert targets to float tensor
#         targets = pt.tensor(train_label_df["correct"].to_numpy(), dtype=pt.float32)

#         # Ensure outputs match target shape
#         outputs = outputs.squeeze()

#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         if verbose and epoch % 10 == 0:
#             print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

#     test_label_df["mlp_classifier_pred"] = (
#         pt.nn.functional.sigmoid(model(activations[test_indices].to(pt.float32)))
#         .detach()
#         .numpy()
#         .squeeze()
#     )

#     mlp_classification_path = experiment_path / "mlp_classifier"
#     mlp_classification_path.mkdir(parents=True, exist_ok=True)
#     evaluate_classification(
#         test_label_df["correct"],
#         test_label_df["mlp_classifier_pred"],
#         experiment_path=mlp_classification_path,
#         extra_info={
#             "model_architecture": str(model),
#             "train_frac": train_frac,
#             "n_epochs": n_epochs,
#             "lr": lr,
#         },  # Note that this isn't intended for logging variation to this classifier, just for logging the parameters used.
#     )


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
    check_logistic_regression_classifier: bool = True,
    # check_mlp_classifier: bool = True,
    train_frac=0.8,
    output_path: str = "./experiment_results",
    notes: str = "",
    _experiment_id: int = None,  # careful, only use this to overwrite existing experiments
) -> None:
    train_frac = float(train_frac)
    # todo: update to save plots and stats to file.

    labels_df, activations = load_labels_df_and_activations(
        layer=layer,
        activations_dir=activations_dir,
        question_answer_file=question_answer_file,
        first_batch=first_batch,
        n_batches=n_batches,
        batch_size=batch_size,
    )
    if not check_correctness_classifier and not check_logistic_regression_classifier:
        raise ValueError(
            "At least one of check_correctness_classifier or check_logistic_regression_classifier must be True"
        )

    ############# DATASET SPECIFIC ###################
    labels_df["correct"] = labels_df["is_correct"]    
    labels_df.loc[labels_df["correct"] == "UK", "correct"] = "True"

    labels_df["special"] = False

    labels_df.loc[labels_df["correct"] == "IDK", "special"] = True
    labels_df.loc[labels_df["correct"] == "IDK", "correct"] = "False"
    # labels_df = labels_df[labels_df["correct"] != "IDK"]
    # activations = activations[list(labels_df.index)]
    # labels_df = labels_df.reset_index(drop=True)


    labels_df["correct"] = labels_df["correct"].map({"True": True, "False": False})

    ############# DATASET SPECIFIC ###################

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
        "check_logistic_regression_classifier": check_logistic_regression_classifier,
        "train_frac": train_frac,
        "timestamp": str(datetime.now()),
        "notes": notes,
    }
    with (experiment_path / "info.json").open("w", encoding="utf-8") as file:
        json.dump(logging_config, file, indent=4, ensure_ascii=False)

    train_indices = np.random.uniform(0, 1, len(labels_df)) < train_frac
    test_indices = ~train_indices
    train_label_df = labels_df.iloc[train_indices]
    test_label_df = labels_df.iloc[test_indices]
    X_train = activations[train_indices]
    X_test = activations[test_indices]

    print(f"{len(X_train)=}")
    print(f"{len(X_test)=}")

    if check_correctness_classifier:
        evaluate_correctness_direction_and_classifier(
            X_train, X_test, train_label_df, test_label_df, experiment_path
        )

    # if check_mlp_classifier:
    #     train_and_evaluate_mlp_classifier(labels_df, activations, experiment_path)
    if check_logistic_regression_classifier:
        evaluate_logistic_regression_classifier(
            X_train, X_test, train_label_df, test_label_df, experiment_path
        )


if __name__ == "__main__":
    print("Run this from the command line at the base level of the repo")

    args_parser = argparse.ArgumentParser()
    # args_parser.add_argument("--config", dest="config", required=True) # is this needed?
    args_parser.add_argument("--model", dest="model", required=True)
    args_parser.add_argument("--layer", dest="layer", required=True)
    args_parser.add_argument("--activations-dir", dest="activations_dir", required=True)
    args_parser.add_argument(
        "--question-answer-file", dest="question_answer_file", required=True
    )
    args_parser.add_argument("--first-batch", dest="first_batch", default=None)
    args_parser.add_argument("--n-batches", dest="n_batches", default=None)
    args_parser.add_argument("--notes", dest="notes", default=None)
    args_parser.add_argument("--random-seed", dest="random_seed", default=42)
    args_parser.add_argument("--train-frac", dest="train_frac", default=0.8)
    args = args_parser.parse_args()

    np.random.seed(args.random_seed)

    classifier_experiment_run(
        model=args.model,
        layer=args.layer,
        activations_dir=args.activations_dir,
        question_answer_file=args.question_answer_file,
        first_batch=int(args.first_batch) if args.first_batch is not None else None,
        n_batches=int(args.n_batches) if args.first_batch is not None else None,
        train_frac=args.train_frac,
        notes=args.notes,
    )
    print("\n\n\nFinished.")
