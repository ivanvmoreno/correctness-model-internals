import argparse
import os
import pandas as pd
import numpy as np
import torch
import xgboost as xgb
import json
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
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Loads and concatenates embeddings and labels for specified datasets/subsets.
    Expects prompt_versions to be a dict mapping each dataset_id to a single prompt version.
    """
    all_X, all_y = [], []
    embedding_dir_name = embedding_model_id.replace("/", "_")

    for dataset_id in dataset_ids:
        prompt_version = prompt_versions.get(dataset_id)
        if not prompt_version:
            logger.warning(f"No prompt version specified for dataset '{dataset_id}'. Skipping this dataset.")
            continue

        for subset in config.datasets[dataset_id].subsets:
            embeddings_path = os.path.join(
                config.base.embeddings_dir,
                embedding_dir_name,
                dataset_id,
                f"{subset}_embeddings.pt"
            )
            labels_path = os.path.join(
                config.base.evaluations_dir,
                model_id,
                dataset_id,
                prompt_version,
                f"{subset}_generations_evaluated.csv"
            )

            if not os.path.exists(embeddings_path) or not os.path.exists(labels_path):
                logger.warning(f"Missing embeddings or labels file for {dataset_id} ({subset}) with prompt version '{prompt_version}'. Skipping.")
                continue

            # --- Load Embeddings ---
            try:
                X_np = torch.load(embeddings_path, weights_only=True).numpy()
                if X_np.shape[0] == 0:
                    continue  # Skip empty files
            except Exception as e:
                logger.warning(f"Error loading embeddings {embeddings_path}: {e}. Skipping.")
                continue

            # --- Load Labels ---
            try:
                labels_df = pd.read_csv(labels_path)
                if label_column not in labels_df.columns:
                    logger.warning(f"Label column '{label_column}' missing in {labels_path}. Skipping.")
                    continue

                y_series = labels_df[label_column]
                # Handle boolean or string 'true'/'false' -> 1/0; treat others/NaNs as 0
                if pd.api.types.is_bool_dtype(y_series):
                    y_processed = y_series.astype(int)
                elif pd.api.types.is_string_dtype(y_series):
                    y_processed = y_series.str.lower().map({'true': 1, 'false': 0})
                else:  # Attempt numeric conversion, otherwise treat as 0
                    y_processed = pd.to_numeric(y_series, errors='coerce')

                # Ensure NaNs become 0 (False)
                y_np = y_processed.fillna(0).astype(int).values

            except Exception as e:
                logger.warning(f"Error loading/processing labels {labels_path}: {e}. Skipping.")
                continue

            # --- Validate Shape and Append ---
            if X_np.shape[0] != y_np.shape[0]:
                logger.warning(f"Shape mismatch for {dataset_id}/{subset}: Embeddings={X_np.shape[0]}, Labels={y_np.shape[0]}. Skipping.")
                continue

            all_X.append(X_np)
            all_y.append(y_np)

    # --- Combine and Final Check ---
    if not all_X:
        logger.error("No data loaded for any specified dataset/subset combination.")
        return None, None

    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    if X_combined.shape[0] == 0:
         logger.error("Combined data is empty after loading.")
         return None, None

    logger.info(f"Total combined data shape loaded: X={X_combined.shape}, y={y_combined.shape}")
    return X_combined, y_combined


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculates standard classification metrics including AUC ROC."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    auc_roc = roc_auc_score(y_true, y_pred_proba)

    return {
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "support": len(y_true)
    }


# --- Main Training and Evaluation Function with Separate Train/Test Datasets ---
def train_evaluate_xgboost(
    config_path: str,
    model_id: str,
    embedding_model_id: str,
    prompt_versions: dict[str, str],
    train_datasets: list[str],
    test_datasets: list[str],
    experiment_name: str,
    random_state: int = 42,
) -> None:
    """
    Loads data from train datasets & test datasets separately (each can have a dedicated prompt version),
    trains XGBoost, evaluates on the test set (incl AUC ROC), and saves results.
    
    The experiment_name is used as a folder/identifier for saving the model and results.
    """
    config = load_config(config_path)
    logger = get_logger("TRAIN_EVAL_XGB_SEPARATE", config.base.log_level)

    logger.info("Starting XGBoost Training & Evaluation (Separate Train/Test)")
    logger.info(f"Model ID: {model_id}, Embedding Model ID: {embedding_model_id}")
    logger.info(f"Train Datasets: {train_datasets}, Test Datasets: {test_datasets}")
    logger.info(f"Prompt Versions: {prompt_versions}")
    logger.info(f"Experiment Name: {experiment_name}")

    # --- 1. Load Training Data ---
    logger.info("Loading training data...")
    X_train, y_train = load_data_for_sets(
        config, train_datasets, embedding_model_id, model_id, prompt_versions, logger
    )

    if X_train is None or y_train is None:
        logger.error("No valid training data. Exiting.")
        return  # Exit if no data loaded for training

    # --- 2. Load Test Data ---
    logger.info("Loading test data...")
    X_test, y_test = load_data_for_sets(
        config, test_datasets, embedding_model_id, model_id, prompt_versions, logger
    )

    if X_test is None or y_test is None:
        logger.error("No valid test data. Exiting.")
        return  # Exit if no data loaded for testing

    logger.info(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # --- 3. XGBoost Model Training ---
    logger.info("Initializing and training XGBoost classifier...")
    xgb_classifier = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=random_state,
        early_stopping_rounds=10
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        xgb_classifier.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False  # Keep logs cleaner
        )
    logger.info("XGBoost training completed.")

    # --- 4. Save Trained Model ---
    output_dir = os.path.join(config.base.classifiers_dir, "assessors", experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    model_filename = f"xgb_classifier_{experiment_name}.json"
    model_save_path = os.path.join(output_dir, model_filename)
    logger.info(f"Saving trained XGBoost model to: {model_save_path}")
    try:
        xgb_classifier.save_model(model_save_path)
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    # --- 5. Evaluation on Test Set ---
    logger.info("Evaluating model on the test set...")
    try:
        y_pred_test = xgb_classifier.predict(X_test)
        y_pred_proba_test = xgb_classifier.predict_proba(X_test)[:, 1]

        test_metrics = calculate_metrics(y_test, y_pred_test, y_pred_proba_test)
        logger.info(
            "Test Set Metrics: "
            f"Accuracy={test_metrics['accuracy']:.4f}, "
            f"AUC ROC={test_metrics['auc_roc']:.4f}, "
            f"F1-weighted={test_metrics['f1_weighted']:.4f}"
        )
    except Exception as e:
        logger.error(f"Error during test prediction/metrics calculation: {e}")
        test_metrics = {"error": f"Prediction/Metrics Error: {e}"}

    # --- 6. Save Results ---
    results = {
        "config": {
            "model_id": model_id,
            "embedding_model_id": embedding_model_id,
            "prompt_versions": prompt_versions,
            "train_datasets": train_datasets,
            "test_datasets": test_datasets,
            "experiment_name": experiment_name,
            "random_state": random_state
        },
        "evaluation_metrics": test_metrics
    }

    output_metrics_path = os.path.join(
        config.base.evaluations_dir, "assessors", experiment_name, "metrics.json"
    )
    output_metrics_dir = os.path.dirname(output_metrics_path)
    os.makedirs(output_metrics_dir, exist_ok=True)

    logger.info(f"Saving evaluation metrics to: {output_metrics_path}")
    try:
        with open(output_metrics_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info("Metrics saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save metrics JSON: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XGBoost training & evaluation with separate lists of train/test datasets."
    )
    parser.add_argument("--config", required=True, help="Path to the main configuration file.")
    parser.add_argument("--model-id", required=True, help="ID of the target model whose answers were evaluated.")
    parser.add_argument("--embedding-model-id", required=True, help="ID of the embedding model used.")
    parser.add_argument(
        "--prompt-versions", 
        default=None,
        help=(
            "Optional: Comma-separated list of dataset:prompt_version pairs "
            "(e.g., 'cities_10k:base,birth_years_4k:base'). "
            "If not provided, 'base' will be used for each dataset."
        )
    )
    parser.add_argument(
        "--train-datasets", 
        required=True, 
        nargs='+', 
        help="List of dataset IDs to use for training."
    )
    parser.add_argument(
        "--test-datasets", 
        required=True, 
        nargs='+', 
        help="List of dataset IDs to use for testing."
    )
    parser.add_argument(
        "--experiment-name", 
        default="unnamed_experiment",
        help="Name for the experiment (used to name files/folders)."
    )
    parser.add_argument(
        "--random-state", 
        type=int, 
        default=42, 
        help="Random seed for XGBoost (default: 42)."
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
    )