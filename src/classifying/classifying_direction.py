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


def calculate_direction_for_group(activations_group, mean, sign):
    return pt.mean(
        sign * (activations_group - mean),
        dim=0,
    )


def calculate_classifying_direction(
    activations: Activations, from_group: Any, to_group: Any, balance=True
):
    # activation = mu + sign * correctness_direction where mu is the mean of all activations (the centroid) and sign is -1 if incorrect and 1 if correct
    # Basically the centroid over the data + the correctness direction (or it flipped) should take you to the centroid of the class.
    from_activations = activations.get_activations(subset=from_group)
    to_activations = activations.get_activations(subset=to_group)

    if balance:
        mean_activations = 0.5 * (
            from_activations.mean(axis=0) + to_activations.mean(axis=0)
        )
        return 0.5 * (
            calculate_direction_for_group(to_activations, mean=mean_activations, sign=1)
            + calculate_direction_for_group(
                from_activations, mean=mean_activations, sign=-1
            )
        )

    all_activations = pt.cat([from_activations, to_activations], axis=0)
    sign = pt.cat(
        [
            np.ones(from_activations.shape[0]),
            -1 * np.ones(to_activations.shape[0]),
        ]
    )
    return calculate_direction_for_group(
        all_activations, mean=all_activations.mean(axis=0), sign=sign
    )
