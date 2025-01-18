import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch as pt
import torch.nn as nn
import torch.optim as optim
from activations import Activations
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, f1_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class DirectionCalculator:
    def __init__(
        self, activations: Activations, from_group: Any, to_group: Any, balance=True
    ):
        self.from_activations = activations.get_activations(subset=from_group)
        self.to_activations = activations.get_activations(subset=to_group)
        self.balance = balance

    @property
    def mean_activations(self):
        if self.balance:
            return 0.5 * (
                self.from_activations.mean(axis=0) + self.to_activations.mean(axis=0)
            )
        return pt.cat([self.from_activations, self.to_activations], axis=0).mean(axis=0)

    @classmethod
    def calculate_direction_for_group(activations_group, mean, sign):
        return pt.mean(
            sign * (activations_group - mean),
            dim=0,
        )

    @property
    def classifying_direction(self):
        # activation = mu + sign * correctness_direction where mu is the mean of all activations (the centroid) and sign is -1 if incorrect and 1 if correct
        # Basically the centroid over the data + the correctness direction (or it flipped) should take you to the centroid of the class.

        if self.balance:
            return 0.5 * (
                self.calculate_direction_for_group(
                    self.to_activations, mean=self.mean_activations, sign=1
                )
                + self.calculate_direction_for_group(
                    self.from_activations, mean=self.mean_activations, sign=-1
                )
            )

        sign = pt.cat(
            [
                np.ones(self.from_activations.shape[0]),
                -1 * np.ones(self.to_activations.shape[0]),
            ]
        )
        return self.calculate_direction_for_group(
            pt.cat([self.from_activations, self.to_activations], axis=0),
            mean=self.mean_activations,
            sign=sign,
        )

    def get_distance_along_classifying_direction(self, tensor: pt.Tensor):
        return (tensor - self.mean_activations) @ self.classifying_direction


class BinaryClassifier:
    def __init__(
        self,
        train_labels,
        train_classification_score,
        test_labels,
        test_classification_score,
        classification_metric_funcs=(accuracy_score, f1_score),
        classification_cut=None,
    ):
        """
        Build a classifier by slicing along a continuous variable on the train set
        """
        if not isinstance(train_labels[0], bool) or not isinstance(
            test_labels[0], bool
        ):
            raise TypeError("Labels must be boolean")

        self.train_labels = train_labels
        self.train_classification_score = train_classification_score
        self.test_labels = test_labels
        self.test_classification_score = test_classification_score

        self.classification_cut = classification_cut
        self.test_pred_class = self.test_classification_score >= (
            self.classification_cut
            if self.classification_cut is not None
            else self.optimal_cut
        )

        (
            self.train_false_positive_rate,
            self.train_true_positive_rate,
            self.train_thresholds,
        ) = roc_curve(self.train_labels, self.train_classification_score)
        self.test_false_positive_rate, self.test_true_positive_rate, _ = roc_curve(
            self.test_labels, self.test_classification_score
        )
        self.roc_auc = auc(self.test_false_positive_rate, self.test_true_positive_rate)
        self.classification_metric_funcs = classification_metric_funcs

        self.classification_metrics = {
            "optimal_cut": self.optimal_cut,
            "roc_auc": float(self.roc_auc),
        }
        for classification_metric in self.classification_metric_funcs:
            self.classification_metrics[classification_metric.__name__] = float(
                classification_metric(self.test_labels, self.test_pred_class)
            )

    @property
    def optimal_cut(self):
        youden_index = self.train_true_positive_rate - self.train_false_positive_rate
        optimal_idx = np.argmax(youden_index[1:]) + 1
        return float(self.train_thresholds[optimal_idx])
