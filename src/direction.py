from typing import Any

import numpy as np

from classifying.activations_handler import ActivationsHandler


class DirectionCalculator:
    def __init__(
        self,
        activations_handler: ActivationsHandler,
        from_group: Any,
        to_group: Any,
        balance=True,
    ):
        self.from_activations = activations_handler.get_activation_groups(
            groups=from_group
        ).activations
        self.to_activations = activations_handler.get_activation_groups(
            groups=to_group
        ).activations
        self.balance = balance

    @property
    def mean_activations(self):
        if self.balance:
            return 0.5 * (
                self.from_activations.mean(axis=0)
                + self.to_activations.mean(axis=0)
            )
        return pt.cat(
            [self.from_activations, self.to_activations], axis=0
        ).mean(axis=0)

    @staticmethod
    def _calculate_direction_for_group(activations_group, mean, sign):
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
                self._calculate_direction_for_group(
                    self.to_activations, mean=self.mean_activations, sign=1
                )
                + self._calculate_direction_for_group(
                    self.from_activations, mean=self.mean_activations, sign=-1
                )
            )

        sign = pt.cat(
            [
                np.ones(self.from_activations.shape[0]),
                -1 * np.ones(self.to_activations.shape[0]),
            ]
        )
        return self._calculate_direction_for_group(
            pt.cat([self.from_activations, self.to_activations], axis=0),
            mean=self.mean_activations,
            sign=sign,
        )

    def get_distance_along_classifying_direction(self, tensor: pt.Tensor):
        return (tensor - self.mean_activations) @ self.classifying_direction
