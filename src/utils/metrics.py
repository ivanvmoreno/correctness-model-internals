from sklearn import metrics

EVAL_METRICS = {
    "accuracy_score": lambda y_true, y_pred: metrics.accuracy_score(y_true, y_pred),
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
