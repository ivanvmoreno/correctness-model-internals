import numpy as np
from sklearn import metrics


def serialize_metrics(metrics):
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    return {k: convert_numpy(v) for k, v in metrics.items()}


EVAL_METRICS = {
    "accuracy_score": lambda y_true, y_pred: metrics.accuracy_score(
        y_true, y_pred
    ),
    "f1_score": lambda y_true, y_pred: metrics.f1_score(
        y_true, y_pred, average="weighted"
    ),
    "precision_score": lambda y_true, y_pred: metrics.precision_score(
        y_true, y_pred, average="weighted"
    ),
    "recall_score": lambda y_true, y_pred: metrics.recall_score(
        y_true, y_pred, average="weighted"
    ),
}
