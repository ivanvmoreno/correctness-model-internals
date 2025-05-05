from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import cupy as cp
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_config
from src.utils.logging import get_logger


def _to_gpu(arr: np.ndarray) -> cp.ndarray:
    """Convert NumPy array → CuPy (GPU) array, keeping dtype and shape."""
    return cp.asarray(arr)


def _parse_boolean_series(
    series: pd.Series, label_column: str, path: str
) -> np.ndarray:
    """Convert heterogenous labels to {0,1}; raise on unknown tokens."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int).values
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).values
    mapping = {"true": 1, "false": 0}
    cleaned = series.astype(str).str.strip().str.lower()
    bad = cleaned[~cleaned.isin(mapping)].unique()
    if bad.size:
        raise ValueError(f"Unexpected tokens in {path}:{bad.tolist()}")
    return cleaned.map(mapping).astype(int).values


def load_data_for_sets(
    config,
    dataset_ids: List[str],
    embedding_model_id: str,
    model_id: str,
    prompt_versions: Dict[str, str],
    logger,
    label_column: str = "correct",
) -> Tuple[np.ndarray | None, np.ndarray | None, Dict[str, Dict[str, int]]]:
    """Load & concatenate embeddings/labels across datasets."""

    all_X, all_y = [], []
    balance: Dict[str, Dict[str, int]] = {}
    emb_dir_name = embedding_model_id.replace("/", "_")

    for ds in dataset_ids:
        pv = prompt_versions.get(ds)
        if not pv or ds not in config.datasets:
            logger.warning("Skipping dataset %s (prompt/version missing)", ds)
            continue
        for subset in config.datasets[ds].subsets:
            emb_path = os.path.join(
                config.base.embeddings_dir,
                emb_dir_name,
                ds,
                f"{subset}_embeddings.pt",
            )
            lab_path = os.path.join(
                config.base.evaluations_dir,
                model_id,
                ds,
                pv,
                f"{subset}_generations_evaluated.csv",
            )
            if not (os.path.exists(emb_path) and os.path.exists(lab_path)):
                logger.warning("Missing file pair for %s/%s", ds, subset)
                continue
            try:
                X = torch.load(emb_path, map_location="cpu", weights_only=True)
                if not isinstance(X, torch.Tensor):
                    raise TypeError
                X_np = X.detach().cpu().numpy().astype(np.float32, copy=False)
                y = _parse_boolean_series(
                    pd.read_csv(lab_path)[label_column], label_column, lab_path
                )
            except Exception as e:
                logger.warning("Failed to load %s/%s: %s", ds, subset, e)
                continue
            if X_np.shape[0] != y.shape[0]:
                logger.warning("Shape mismatch %s/%s", ds, subset)
                continue
            all_X.append(X_np)
            all_y.append(y)
            for val, cnt in zip(*np.unique(y, return_counts=True)):
                balance.setdefault(ds, {}).setdefault(str(val), 0)
                balance[ds][str(val)] += int(cnt)
    if not all_X:
        return None, None, {}
    return np.concatenate(all_X), np.concatenate(all_y), balance


def _metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, Any]:
    res: Dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "support": len(y_true),
    }
    for avg in ("weighted", "macro"):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        res.update({f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f1})
    if len(np.unique(y_true)) > 1:
        try:
            res["auc_roc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            res["auc_roc"] = None
    else:
        res["auc_roc"] = None
    return res


def _make_classifier(
    classifier_type: str,
    random_state: int,
    spw: float,
    *,
    n_estimators: int | None = None,
):
    """Factory that returns a fresh classifier instance for a fold/final fit."""
    if classifier_type == "xgboost":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=random_state,
            tree_method="hist",
            device="cuda",
            n_jobs=-1,
            n_estimators=n_estimators if n_estimators else 1000,
            early_stopping_rounds=10 if n_estimators is None else None,
            scale_pos_weight=spw,
        )
    elif classifier_type == "logistic":
        base = LogisticRegression(
            random_state=random_state,
            solver="saga",
            max_iter=300,
            tol=1e-4,
            n_jobs=-1,
            class_weight="balanced",
        )
        return Pipeline([("scaler", StandardScaler()), ("logreg", base)])
    else:
        raise ValueError(f"Unsupported clf {classifier_type}")


def train_evaluate_classifier(
    *,
    config_path: str,
    model_id: str,
    embedding_model_id: str,
    classifier_type: str,
    prompt_versions: Dict[str, str],
    train_datasets: List[str],
    test_datasets: List[str],
    experiment_name: str,
    random_state: int = 42,
    top_k: Optional[int] = None,
    cv_folds: int = 5,
    holdout_size: int = 0,
) -> None:
    """Train, cross‑validate, and evaluate a classifier.

    Args:
    holdout_size: If >0, reserve the first n samples (after concatenation, before shuffling)
        from the training datasets as an in‑distribution held‑out set
    """

    config = load_config(config_path)
    logger = get_logger(
        f"ASSESSOR_{classifier_type.upper()}", config.base.log_level
    )

    # Load training data
    X_all, y_all, train_balance = load_data_for_sets(
        config,
        train_datasets,
        embedding_model_id,
        model_id,
        prompt_versions,
        logger,
    )
    if X_all is None:
        logger.error("No training data; abort.")
        return

    if top_k and 0 < top_k < X_all.shape[1]:
        X_all = X_all[:, :top_k]

    # Holdout set
    held_X: Optional[np.ndarray]
    held_y: Optional[np.ndarray]

    if holdout_size and holdout_size > 0:
        holdout_size = min(holdout_size, len(X_all))
        held_X, held_y = X_all[:holdout_size], y_all[:holdout_size]
        X_train_full, y_train_full = X_all[holdout_size:], y_all[holdout_size:]
        logger.info(
            "Reserved %d samples for held‑out in‑distribution evaluation (%.2f of data)",
            holdout_size,
            holdout_size / len(X_all),
        )
    else:
        held_X = held_y = None
        X_train_full, y_train_full = X_all, y_all

    # K-fold cross-validation
    skf = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )
    oof_pred = np.zeros_like(y_train_full, dtype=int)
    oof_proba = np.zeros_like(y_train_full, dtype=float)

    fold_metrics: List[Dict[str, Any]] = []
    best_iters: List[int] = []  # collect best_iteration per fold

    logger.info(
        "Starting %d‑fold CV (training portion: %d samples)",
        cv_folds,
        len(X_train_full),
    )
    for k, (train_idx, val_idx) in enumerate(
        skf.split(X_train_full, y_train_full), 1
    ):
        # Fold data
        X_tr, y_tr = X_train_full[train_idx], y_train_full[train_idx]
        X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

        # Class balance for this fold
        pos_fold, neg_fold = (y_tr == 1).sum(), (y_tr == 0).sum()
        spw_fold = neg_fold / pos_fold if pos_fold else 1.0

        clf = _make_classifier(classifier_type, random_state, spw_fold)

        if classifier_type == "xgboost":
            X_tr_gpu = _to_gpu(X_tr)
            X_val_gpu = _to_gpu(X_val)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                clf.fit(
                    X_tr_gpu, y_tr, eval_set=[(X_val_gpu, y_val)], verbose=False
                )
            if (
                hasattr(clf, "best_iteration")
                and clf.best_iteration is not None
            ):
                best_iters.append(clf.best_iteration + 1)
            yhat = clf.predict(X_val_gpu)
            yproba = clf.predict_proba(X_val_gpu)[:, 1]
        else:
            clf.fit(X_tr, y_tr)
            yhat = clf.predict(X_val)
            yproba = clf.predict_proba(X_val)[:, 1]

        oof_pred[val_idx] = yhat
        oof_proba[val_idx] = yproba
        m = _metrics(y_val, yhat, yproba)
        fold_metrics.append(m)
        logger.info(
            "Fold %d | AUC=%.3f | F1w=%.3f",
            k,
            m.get("auc_roc", 0) or 0,
            m["f1_weighted"],
        )

    cv_summary = _metrics(y_train_full, oof_pred, oof_proba)
    logger.info(
        "CV mean AUC=%.3f, Accuracy=%.3f",
        cv_summary.get("auc_roc", 0) or 0,
        cv_summary["accuracy"],
    )

    # Best hyperparameters
    if classifier_type == "xgboost" and best_iters:
        n_estimators_final = int(round(float(np.mean(best_iters))))
        logger.info("Using n_estimators=%d for final model", n_estimators_final)
    else:
        n_estimators_final = None  # logistic or fallback

    # Final fit
    pos, neg = (y_train_full == 1).sum(), (y_train_full == 0).sum()
    spw_full = neg / pos if pos else 1.0

    final_clf = _make_classifier(
        classifier_type,
        random_state,
        spw_full,
        n_estimators=n_estimators_final,
    )

    if classifier_type == "xgboost":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            final_clf.fit(_to_gpu(X_train_full), y_train_full, verbose=False)
    else:
        final_clf.fit(X_train_full, y_train_full)

    # Evaluation on test datasets
    X_test, y_test, test_balance = load_data_for_sets(
        config,
        test_datasets,
        embedding_model_id,
        model_id,
        prompt_versions,
        logger,
    )
    if X_test is not None:
        if top_k and top_k < X_test.shape[1]:
            X_test = X_test[:, :top_k]
        if classifier_type == "xgboost":
            X_test_gpu = _to_gpu(X_test.astype(np.float32, copy=False))
            yhat = final_clf.predict(X_test_gpu)
            yproba = final_clf.predict_proba(X_test_gpu)[:, 1]
        else:
            yhat = final_clf.predict(X_test.astype(np.float32, copy=False))
            yproba = final_clf.predict_proba(
                X_test.astype(np.float32, copy=False)
            )[:, 1]
    else:
        yhat = yproba = None
        test_balance = {}

    eval_metrics: Dict[str, Any] = {
        "cross_validation": cv_summary,
        "fold_metrics": fold_metrics,
    }

    if yhat is not None:
        eval_metrics["aggregated"] = _metrics(y_test, yhat, yproba)
        eval_metrics["aggregated"]["data_balance"] = test_balance
    else:
        eval_metrics["aggregated"] = {"status": "skipped"}

    # Evaluation on holdout set
    if held_X is not None and len(held_X) > 0:
        if classifier_type == "xgboost":
            held_gpu = _to_gpu(held_X.astype(np.float32, copy=False))
            yh = final_clf.predict(held_gpu)
            yp = final_clf.predict_proba(held_gpu)[:, 1]
        else:
            yh = final_clf.predict(held_X.astype(np.float32, copy=False))
            yp = final_clf.predict_proba(held_X.astype(np.float32, copy=False))[
                :, 1
            ]
        eval_metrics["held_out"] = {
            "n_samples": len(held_X),
            **_metrics(held_y, yh, yp),
        }
    else:
        eval_metrics["held_out"] = {"status": "not_requested"}

    # Per‑dataset evaluation
    by_ds: Dict[str, Any] = {}
    for ds in test_datasets:
        X_ds, y_ds, bal = load_data_for_sets(
            config,
            [ds],
            embedding_model_id,
            model_id,
            prompt_versions,
            logger,
        )
        if X_ds is None:
            by_ds[ds] = {"status": "skipped"}
            continue
        if top_k and top_k < X_ds.shape[1]:
            X_ds = X_ds[:, :top_k]
        if classifier_type == "xgboost":
            X_ds_gpu = _to_gpu(X_ds.astype(np.float32, copy=False))
            yhat_ds = final_clf.predict(X_ds_gpu)
            yproba_ds = final_clf.predict_proba(X_ds_gpu)[:, 1]
        else:
            yhat_ds = final_clf.predict(X_ds.astype(np.float32, copy=False))
            yproba_ds = final_clf.predict_proba(
                X_ds.astype(np.float32, copy=False)
            )[:, 1]
        by_ds[ds] = _metrics(y_ds, yhat_ds, yproba_ds)
        by_ds[ds]["data_balance"] = bal.get(ds, {})
    eval_metrics["by_dataset"] = by_ds

    # Save model
    out_dir = os.path.join(
        config.base.classifiers_dir, "assessors", experiment_name
    )
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(
        out_dir,
        f"{classifier_type}_classifier_{experiment_name}.{ 'json' if classifier_type=='xgboost' else 'joblib'}",
    )
    try:
        if classifier_type == "xgboost":
            final_clf.save_model(model_path)
        else:
            joblib.dump(final_clf, model_path)
    except Exception as e:
        logger.error("Failed saving model: %s", e)

    result = {
        "config": {
            "model_id": model_id,
            "embedding_model_id": embedding_model_id,
            "classifier_type": classifier_type,
            "prompt_versions": prompt_versions,
            "train_datasets": train_datasets,
            "test_datasets": test_datasets,
            "experiment_name": experiment_name,
            "random_state": random_state,
            "top_k": top_k,
            "cv_folds": cv_folds,
            "n_estimators_final": n_estimators_final,
            "holdout_size": holdout_size,
        },
        "evaluation_metrics": eval_metrics,
        "initial_data_balance": {
            "train": train_balance,
            "test_aggregated": test_balance,
        },
    }
    metrics_path = os.path.join(
        config.base.evaluations_dir,
        "assessors",
        experiment_name,
        f"{classifier_type}_metrics.json",
    )
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Wrote metrics → %s", metrics_path)


def _parse_pv(arg: str | None) -> Dict[str, str]:
    if not arg:
        return {}
    out = {}
    for pair in arg.split(","):
        if ":" not in pair:
            raise ValueError(f"Bad prompt‑version pair: {pair}")
        ds, pv = pair.split(":", 1)
        out[ds.strip()] = pv.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--embedding-model-id", required=True)
    ap.add_argument(
        "--classifier-type", choices=["xgboost", "logistic"], default="xgboost"
    )
    ap.add_argument("--prompt-versions", default=None)
    ap.add_argument("--train-datasets", required=True, nargs="+")
    ap.add_argument("--test-datasets", required=True, nargs="+")
    ap.add_argument("--experiment-name", default="unnamed_experiment")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument(
        "--cv-folds", type=int, default=5, help="Number of CV folds (default 5)"
    )
    ap.add_argument(
        "--holdout-size",
        type=int,
        default=0,
        help="Reserve the first N samples from the concatenated training data as a held‑out set. 0 disables the behaviour.",
    )
    args = ap.parse_args()

    pv = _parse_pv(args.prompt_versions)
    for ds in set(args.train_datasets + args.test_datasets):
        pv.setdefault(ds, "base")

    train_evaluate_classifier(
        config_path=args.config,
        model_id=args.model_id,
        embedding_model_id=args.embedding_model_id,
        classifier_type=args.classifier_type,
        prompt_versions=pv,
        train_datasets=args.train_datasets,
        test_datasets=args.test_datasets,
        experiment_name=args.experiment_name,
        random_state=args.random_state,
        top_k=args.top_k,
        cv_folds=args.cv_folds,
        holdout_size=args.holdout_size,
    )


if __name__ == "__main__":
    main()
