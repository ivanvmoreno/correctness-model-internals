from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from .activations_handler import ActivationsHandler
from .direction_calculator import DirectionCalculator
from .typing import BatchValues


class BinaryClassifier:
    """
    A binary classifier that can be used to classify data into two classes based on
    a classification score.

    Parameters
    ----------
    classification_metrics : dict[str, float]
        Classification metrics for test data. This is the main purpose of this class.
    train_labels : pd.Series
        Boolean labels for training data
    train_classification_score : BatchValues
        Classification scores for training data
    test_labels : pd.Series
        Boolean labels for test data
    test_classification_score : BatchValues
        Classification scores for test data
    train_roc_fprs : np.ndarray
        False positive rates for training data given thresholds
    train_roc_tprs : np.ndarray
        True positive rates for training data given thresholds
    train_roc_thresholds : np.ndarray
        Thresholds for training data roc curve
    test_roc_fprs : np.ndarray
        False positive rates for test data given thresholds
    test_roc_tprs : np.ndarray
        True positive rates for test data given thresholds
    test_roc_thresholds : np.ndarray
        Thresholds for test data roc curve
    test_roc_auc : float
        Area under the ROC curve for test data
    classification_metric_funcs : tuple[Callable, ...]
        Functions to calculate classification metrics
    optimal_cut : float
        Optimal cut based on training data (or classification_cut if provided)
    test_pred_class : np.ndarray
        Predicted classes for test data
    """

    def __init__(
        self,
        train_labels: pd.Series,
        train_classification_score: BatchValues,
        test_labels: pd.Series,
        test_classification_score: BatchValues,
        classification_metric_funcs: tuple[Callable, ...] = (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        ),
        classification_cut: None | float = None,
    ):
        if not train_labels.dtype == "bool" or not test_labels.dtype == "bool":
            raise TypeError("Labels must be boolean")

        assert len(train_labels), "can't be empty"
        assert len(test_labels), "can't be empty"
        assert (
            len(train_labels) == train_classification_score.shape[0]
        ), "must have the same size"
        assert (
            len(test_labels) == test_classification_score.shape[0]
        ), "must have the same size"

        self.train_labels = train_labels
        self.train_classification_score = train_classification_score
        self.test_labels = test_labels
        self.test_classification_score = test_classification_score

        (
            self.train_roc_fprs,
            self.train_roc_tprs,
            self.train_roc_thresholds,
        ) = roc_curve(self.train_labels, self.train_classification_score)
        self.test_roc_fprs, self.test_roc_tprs, _ = roc_curve(
            self.test_labels, self.test_classification_score
        )
        self.test_roc_auc = float(auc(self.test_roc_fprs, self.test_roc_tprs))
        self.classification_metric_funcs = classification_metric_funcs

        self.optimal_cut = (
            classification_cut
            if classification_cut is not None
            else self.optimal_train_set_cut
        )
        self.test_pred_class = self.test_classification_score >= (
            self.optimal_cut
        )

        self.classification_metrics = {
            "optimal_cut": self.optimal_cut,
            "optimal_train_set_cut": self.optimal_train_set_cut,
            "test_roc_auc": float(self.test_roc_auc),
        }
        for classification_metric in self.classification_metric_funcs:
            self.classification_metrics[classification_metric.__name__] = float(
                classification_metric(self.test_labels, self.test_pred_class)
            )

    @property
    def optimal_train_set_cut(self) -> float:
        """
        Calculate the optimal cut along the classification scores.
        Done on the training set by maximizing the difference between the true positive
        rate and false positive rate.

        Returns
        -------
        float
            The optimal cut on the training set
        """
        youden_index = self.train_roc_tprs - self.train_roc_fprs
        optimal_idx = np.argmax(youden_index[1:]) + 1
        return float(self.train_roc_thresholds[optimal_idx])


def get_correctness_direction_classifier(
    activations_handler_train: ActivationsHandler,
    activations_handler_test: ActivationsHandler,
) -> tuple[BinaryClassifier, DirectionCalculator]:
    """
    Build a classifier that uses the directions in activation space between groups
    of activations.

    Parameters
    ----------
    activations_handler_train : ActivationsHandler
        Activations handler for training data
    activations_handler_test : ActivationsHandler
        Activations handler for test data

    Returns
    -------
    tuple[BinaryClassifier, DirectionCalculator]
        The classifier and direction calculator
    """
    direction_calculator = DirectionCalculator(
        activations_from=activations_handler_train.get_groups(
            False
        ).activations,
        activations_to=activations_handler_train.get_groups(True).activations,
    )
    direction_classifier = BinaryClassifier(
        train_labels=activations_handler_train.labels,
        train_classification_score=direction_calculator.get_distance_along_classifying_direction(
            activations_handler_train.activations
        ),
        test_labels=activations_handler_test.labels,
        test_classification_score=direction_calculator.get_distance_along_classifying_direction(
            activations_handler_test.activations
        ),
    )
    return direction_classifier, direction_calculator


def get_logistic_regression_classifier(
    activations_handler_train: ActivationsHandler,
    activations_handler_test: ActivationsHandler,
    classification_cut: float = 0.5,
    scaler_model_tuple: tuple | None = None,
) -> tuple[BinaryClassifier, LogisticRegression]:
    """
    Build a logistic regression classifier that uses the activations as features.

    Parameters
    ----------
    activations_handler_train : ActivationsHandler
        Activations handler for training data
    activations_handler_test : ActivationsHandler
        Activations handler for test data
    classification_cut : float
        The cut to use for classification, defaults to 0.5

    Returns
    -------
    tuple[BinaryClassifier, LogisticRegression]
        The logistic regression classifier and LR model
    """
    if scaler_model_tuple is None:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(activations_handler_train.activations)
        model = LogisticRegression(
            random_state=42,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
        )
        model.fit(X_train, activations_handler_train.labels)
    else:
        scaler, model = scaler_model_tuple
        X_train = scaler.transform(activations_handler_train.activations)
    X_test = scaler.transform(activations_handler_test.activations)

    logistic_regression_classifier = BinaryClassifier(
        train_labels=activations_handler_train.labels,
        train_classification_score=model.predict_proba(X_train)[:, 1],
        test_labels=activations_handler_test.labels,
        test_classification_score=model.predict_proba(X_test)[:, 1],
        classification_cut=classification_cut,
    )
    return logistic_regression_classifier, (scaler, model)


def get_between_class_variance_and_within_class_variance(
    ah: ActivationsHandler, groups: tuple = (False, True)
):
    """
    Calculate the between class variance and within class variance of the activations.

    Parameters
    ----------
    ah : ActivationsHandler
        The activations handler to calculate the variances for
    groups : tuple
        The groups labels to calculate the variances for

    Returns
    -------
    tuple[float, float]
        The between class variance and within class variance
    """
    global_mean = ah.activations.mean(dim=0)

    between_class_variance, within_class_variance = 0.0, 0.0
    for group in groups:
        group_activations = ah.get_groups(group).activations
        group_mean = group_activations.mean(dim=0)
        group_weight = group_activations.shape[0] / ah.activations.shape[0]

        group_diff = group_mean - global_mean
        between_class_variance += (group_weight * (group_diff**2).sum()).item()

        within_class_variance += (
            group_weight
            * ((group_activations - group_mean) ** 2).sum(dim=1).mean()
        ).item()

    if within_class_variance < 1e-8:
        return float("inf")

    return between_class_variance, within_class_variance
