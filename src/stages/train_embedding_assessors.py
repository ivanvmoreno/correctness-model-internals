import argparse
import os
import pandas as pd
import numpy as np
import torch
import xgboost as xgb
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import warnings

from src.utils.config import load_config
from src.utils.logging import get_logger


def load_data_for_sets(
    config,
    dataset_ids: list[str],
    embedding_model_id: str,
    model_id: str,
    prompt_versions: dict[str, str],
    logger,
    label_column: str = "correct",
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """
    Loads and concatenates embeddings and labels for specified datasets/subsets.
    Expects prompt_versions to be a dict mapping each dataset_id to a single prompt version.

    Also accumulates and returns class balance statistics per dataset as a dict.
    """
    all_X, all_y = [], []
    # Initialize dictionary to store class counts per dataset.
    balance_dict = {}

    embedding_dir_name = embedding_model_id.replace("/", "_")

    for dataset_id in dataset_ids:
        prompt_version = prompt_versions.get(dataset_id)
        if not prompt_version:
            logger.warning(
                f"No prompt version specified for dataset '{dataset_id}'. Skipping this dataset."
            )
            continue

        for subset in config.datasets[dataset_id].subsets:
            embeddings_path = os.path.join(
                config.base.embeddings_dir,
                embedding_dir_name,
                dataset_id,
                f"{subset}_embeddings.pt",
            )
            logger.info(f"Loading embeddings from: {embeddings_path}")
            labels_path = os.path.join(
                config.base.evaluations_dir,
                model_id,
                dataset_id,
                prompt_version,
                f"{subset}_generations_evaluated.csv",
            )
            logger.info(f"Loading labels from: {labels_path}")

            if not os.path.exists(embeddings_path) or not os.path.exists(
                labels_path
            ):
                logger.warning(
                    f"Missing embeddings or labels file for {dataset_id} ({subset}) with prompt version '{prompt_version}'. Skipping."
                )
                continue

            try:
                X_np = torch.load(embeddings_path, weights_only=True).numpy()
                if X_np.shape[0] == 0:
                    continue 
            except Exception as e:
                logger.warning(
                    f"Error loading embeddings {embeddings_path}: {e}. Skipping."
                )
                continue

            try:
                labels_df = pd.read_csv(labels_path)
                if label_column not in labels_df.columns:
                    logger.warning(
                        f"Label column '{label_column}' missing in {labels_path}. Skipping."
                    )
                    continue

                y_series = labels_df[label_column]
                # Handle boolean or string 'true'/'false' -> 1/0; treat others/NaNs as 0
                if pd.api.types.is_bool_dtype(y_series):
                    y_processed = y_series.astype(int)
                elif pd.api.types.is_string_dtype(y_series):
                    y_processed = y_series.str.lower().map(
                        {"true": 1, "false": 0}
                    )
                else:  # Attempt numeric conversion, otherwise treat as 0
                    y_processed = pd.to_numeric(y_series, errors="coerce")

                # Ensure NaNs become 0 (False)
                y_np = y_processed.fillna(0).astype(int).values

            except Exception as e:
                logger.warning(
                    f"Error loading/processing labels {labels_path}: {e}. Skipping."
                )
                continue

            if X_np.shape[0] != y_np.shape[0]:
                logger.warning(
                    f"Shape mismatch for {dataset_id}/{subset}: Embeddings={X_np.shape[0]}, Labels={y_np.shape[0]}. Skipping."
                )
                continue

            if dataset_id not in balance_dict:
                balance_dict[dataset_id] = {}
            unique_labels, counts = np.unique(y_np, return_counts=True)
            for label, count in zip(unique_labels, counts):
                label_str = str(label)
                balance_dict[dataset_id][label_str] = balance_dict[
                    dataset_id
                ].get(label_str, 0) + int(count)

            all_X.append(X_np)
            all_y.append(y_np)

    if not all_X:
        logger.error(
            "No data loaded for any specified dataset/subset combination."
        )
        return None, None, {}

    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    if X_combined.shape[0] == 0:
        logger.error("Combined data is empty after loading.")
        return None, None, {}

    logger.info(
        f"Total combined data shape loaded: X={X_combined.shape}, y={y_combined.shape}"
    )
    return X_combined, y_combined, balance_dict


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculates standard classification metrics including AUC ROC."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    auc_roc = roc_auc_score(y_true, y_pred_proba)

    return {
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "support": len(y_true),
    }


def train_evaluate_xgboost(
    config_path: str,
    model_id: str,
    embedding_model_id: str,
    prompt_versions: dict[str, str],
    train_datasets: list[str],
    test_datasets: list[str],
    experiment_name: str,
    random_state: int = 42,
    top_k: int | None = None,
) -> None:
    """
    Loads data from train datasets & test datasets separately (each can have a dedicated prompt version),
    trains XGBoost, evaluates on the test set (with breakdown by individual test dataset as well as
    aggregated performance), and saves results.

    The experiment_name is used as a folder/identifier for saving the model and results.

    If top_k is provided, only the top-K dimensions of the input embeddings are used.
    """
    config = load_config(config_path)
    logger = get_logger("TRAIN_EVAL_XGB_SEPARATE", config.base.log_level)

    logger.info("Starting XGBoost Training & Evaluation (Separate Train/Test)")
    logger.info(
        f"Model ID: {model_id}, Embedding Model ID: {embedding_model_id}"
    )
    logger.info(
        f"Train Datasets: {train_datasets}, Test Datasets: {test_datasets}"
    )
    logger.info(f"Prompt Versions: {prompt_versions}")
    logger.info(f"Experiment Name: {experiment_name}")
    if top_k is not None:
        logger.info(
            f"Using only the top {top_k} dimensions of the embeddings for training and evaluation."
        )

    logger.info("Loading training data...")
    X_train, y_train, train_balance = load_data_for_sets(
        config,
        train_datasets,
        embedding_model_id,
        model_id,
        prompt_versions,
        logger,
    )

    if X_train is None or y_train is None:
        logger.error("No valid training data. Exiting.")
        return  # Exit if no data loaded for training

    # Slice training embeddings if top_k is provided and valid
    if top_k is not None:
        if top_k < X_train.shape[1]:
            X_train = X_train[:, :top_k]
        else:
            logger.warning(
                f"Requested top_k ({top_k}) is greater than or equal to available dimensions ({X_train.shape[1]}). Using full embeddings."
            )

    X_train_main, X_in_dist, y_train_main, y_in_dist = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=random_state,
        stratify=y_train,
    )

    unique_labels, counts = np.unique(y_in_dist, return_counts=True)
    in_dist_balance = {
        str(label): int(count) for label, count in zip(unique_labels, counts)
    }

    logger.info(
        f"In-distribution hold-out: X={X_in_dist.shape}, balance={in_dist_balance}"
    )

    logger.info("Loading aggregated test data...")
    X_test, y_test, test_balance = load_data_for_sets(
        config,
        test_datasets,
        embedding_model_id,
        model_id,
        prompt_versions,
        logger,
    )

    if X_test is None or y_test is None:
        logger.error("No valid test data. Exiting.")
        return  # Exit if no data loaded for testing

    # Slice aggregated test embeddings if top_k is provided
    if top_k is not None:
        if top_k < X_test.shape[1]:
            X_test = X_test[:, :top_k]
        else:
            logger.warning(
                f"Requested top_k ({top_k}) is greater than or equal to available test dimensions ({X_test.shape[1]}). Using full embeddings."
            )

    logger.info(
        f"Train data shape: {X_train_main.shape}, Aggregated test data shape: {X_test.shape}"
    )

    logger.info("Initializing and training XGBoost classifier...")
    xgb_classifier = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=random_state,
        early_stopping_rounds=10,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        xgb_classifier.fit(
            X_train_main,
            y_train_main,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
    logger.info("XGBoost training completed.")

    output_dir = os.path.join(
        config.base.classifiers_dir, "assessors", experiment_name
    )
    os.makedirs(output_dir, exist_ok=True)

    model_filename = f"xgb_classifier_{experiment_name}.json"
    model_save_path = os.path.join(output_dir, model_filename)
    logger.info(f"Saving trained XGBoost model to: {model_save_path}")
    try:
        xgb_classifier.save_model(model_save_path)
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    logger.info("Evaluating model on the aggregated test set...")
    try:
        y_pred_test = xgb_classifier.predict(X_test)
        y_pred_proba_test = xgb_classifier.predict_proba(X_test)[:, 1]

        aggregated_metrics = calculate_metrics(
            y_test, y_pred_test, y_pred_proba_test
        )
        logger.info(
            "Aggregated Test Set Metrics: "
            f"Accuracy={aggregated_metrics['accuracy']:.4f}, "
            f"AUC ROC={aggregated_metrics['auc_roc']:.4f}, "
            f"F1-weighted={aggregated_metrics['f1_weighted']:.4f}"
        )
    except Exception as e:
        logger.error(
            f"Error during aggregated test prediction/metrics calculation: {e}"
        )
        aggregated_metrics = {"error": f"Prediction/Metrics Error: {e}"}

    logger.info("Evaluating in-distribution hold-out slice...")
    y_pred_indist = xgb_classifier.predict(X_in_dist)
    y_pred_proba_indist = xgb_classifier.predict_proba(X_in_dist)[:, 1]
    in_dist_metrics = calculate_metrics(
        y_in_dist, y_pred_indist, y_pred_proba_indist
    )

    individual_metrics = {}
    logger.info("Evaluating model on individual test datasets...")
    for dataset_id in test_datasets:
        logger.info(f"Loading test data for dataset: {dataset_id}")
        X_indiv, y_indiv, indiv_balance = load_data_for_sets(
            config,
            [dataset_id],
            embedding_model_id,
            model_id,
            prompt_versions,
            logger,
        )
        if X_indiv is None or y_indiv is None:
            logger.warning(
                f"Skipping individual metrics for test dataset {dataset_id} due to missing data."
            )
            continue

        # Slice individual test embeddings if top_k is provided
        if top_k is not None:
            if top_k < X_indiv.shape[1]:
                X_indiv = X_indiv[:, :top_k]
            else:
                logger.warning(
                    f"Requested top_k ({top_k}) is greater than or equal to available dimensions for dataset {dataset_id} ({X_indiv.shape[1]}). Using full embeddings."
                )

        try:
            y_pred_indiv = xgb_classifier.predict(X_indiv)
            y_pred_proba_indiv = xgb_classifier.predict_proba(X_indiv)[:, 1]
            metrics_indiv = calculate_metrics(
                y_indiv, y_pred_indiv, y_pred_proba_indiv
            )
            individual_metrics[dataset_id] = {
                "metrics": metrics_indiv,
                "data_balance": indiv_balance.get(dataset_id, {}),
            }
            logger.info(
                f"Dataset {dataset_id} Metrics: Accuracy={metrics_indiv['accuracy']:.4f}, "
                f"AUC ROC={metrics_indiv['auc_roc']:.4f}, F1-weighted={metrics_indiv['f1_weighted']:.4f}"
            )
        except Exception as e:
            logger.error(
                f"Error during evaluation for dataset {dataset_id}: {e}"
            )
            individual_metrics[dataset_id] = {
                "error": f"Prediction/Metrics Error: {e}"
            }

    results = {
        "config": {
            "model_id": model_id,
            "embedding_model_id": embedding_model_id,
            "prompt_versions": prompt_versions,
            "train_datasets": train_datasets,
            "test_datasets": test_datasets,
            "experiment_name": experiment_name,
            "random_state": random_state,
            "top_k": top_k,
        },
        "evaluation_metrics": {
            "in_distribution": {
                "metrics": in_dist_metrics,
                "data_balance": in_dist_balance,
            },
            "aggregated": aggregated_metrics,
            "by_dataset": individual_metrics,
        },
        "data_balance": {"train": train_balance, "test": test_balance},
    }

    output_metrics_path = os.path.join(
        config.base.evaluations_dir,
        "assessors",
        experiment_name,
        "metrics.json",
    )
    output_metrics_dir = os.path.dirname(output_metrics_path)
    os.makedirs(output_metrics_dir, exist_ok=True)

    logger.info(f"Saving evaluation metrics to: {output_metrics_path}")
    try:
        with open(output_metrics_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info("Metrics saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save metrics JSON: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XGBoost training & evaluation with separate lists of train/test datasets."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the main configuration file."
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="ID of the target model whose answers were evaluated.",
    )
    parser.add_argument(
        "--embedding-model-id",
        required=True,
        help="ID of the embedding model used.",
    )
    parser.add_argument(
        "--prompt-versions",
        default=None,
        help=(
            "Optional: Comma-separated list of dataset:prompt_version pairs "
            "(e.g., 'cities_10k:base,birth_years_4k:base'). "
            "If not provided, 'base' will be used for each dataset."
        ),
    )
    parser.add_argument(
        "--train-datasets",
        required=True,
        nargs="+",
        help="List of dataset IDs to use for training.",
    )
    parser.add_argument(
        "--test-datasets",
        required=True,
        nargs="+",
        help="List of dataset IDs to use for testing.",
    )
    parser.add_argument(
        "--experiment-name",
        default="unnamed_experiment",
        help="Name for the experiment (used to name files/folders).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for XGBoost (default: 42).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="If provided, use only the top K dimensions of the input embeddings.",
    )

    args = parser.parse_args()

    # Parse prompt-versions into a dictionary
    prompt_versions_dict = {}
    if args.prompt_versions:
        for pair in args.prompt_versions.split(","):
            if ":" in pair:
                ds, pv = pair.split(":", 1)
                prompt_versions_dict[ds.strip()] = pv.strip()

    # Default to 'base' if not in prompt_versions_dict
    for ds in set(args.train_datasets + args.test_datasets):
        if ds not in prompt_versions_dict:
            prompt_versions_dict[ds] = "base"

    train_evaluate_xgboost(
        config_path=args.config,
        model_id=args.model_id,
        embedding_model_id=args.embedding_model_id,
        prompt_versions=prompt_versions_dict,
        train_datasets=args.train_datasets,
        test_datasets=args.test_datasets,
        experiment_name=args.experiment_name,
        random_state=args.random_state,
        top_k=args.top_k,
    )
