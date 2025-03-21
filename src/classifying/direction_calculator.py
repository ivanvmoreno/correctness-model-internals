import torch as pt

from .typing import ActivationsVector, BatchActivationsVector, BatchValues


class DirectionCalculator:
    """
    Utilities for calculating the direction between two groups of activations.

    Attributes
    ----------
    activations_from: BatchActivationsVector
        Activations from the first group
    activations_to: BatchActivationsVector
        Activations from the second group
    balance: bool
        Whether to weight the two groups equally
    centroid_from: ActivationsVector
        The centroid of the first group
    centroid_to: ActivationsVector
        The centroid of the second group
    max_activations_from: ActivationsVector
        The maximum of each activation dimension for the first group
    min_activations_from: ActivationsVector
        The minimum of each activation dimension for the first group
    max_activations_to: ActivationsVector
        The maximum of each activation dimension for the second group
    min_activations_to: ActivationsVector
        The minimum of each activation dimension for the second group
    mean_activations: ActivationsVector
        The mean of all activations
    classifying_direction: ActivationsVector
        The classifying direction
    """

    def __init__(
        self,
        activations_from: BatchActivationsVector,
        activations_to: BatchActivationsVector,
        balance: bool = True,
    ):
        """
        Initialize the DirectionCalculator.

        Parameters
        ----------
        activations_from: BatchActivationsVector
            Activations from the first group
        activations_to: BatchActivationsVector
            Activations from the second group
        balance: bool, default=True
            Whether to weight the two groups equally
        """
        self.activations_from = activations_from
        self.activations_to = activations_to
        self.balance = balance

        self.centroid_from = self.activations_from.mean(dim=0)
        self.centroid_to = self.activations_to.mean(dim=0)
        self.max_activations_from = self.activations_from.max(dim=0).values
        self.min_activations_from = self.activations_from.min(dim=0).values
        self.max_activations_to = self.activations_to.max(dim=0).values
        self.min_activations_to = self.activations_to.min(dim=0).values

    @property
    def mean_activations(self) -> ActivationsVector:
        """
        Calculate the mean of all activations.

        If balance is True, returns the mean of the two group centroids.
        If balance is False, returns the mean of all activations combined.

        Returns
        -------
        ActivationsVector
            The mean activations vector
        """
        if self.balance:
            return 0.5 * (self.centroid_from + self.centroid_to)
        return pt.cat([self.activations_from, self.activations_to], dim=0).mean(
            dim=0
        )

    @staticmethod
    def _calculate_direction_for_group(
        activations_group: BatchActivationsVector,
        mean: ActivationsVector,
        sign: BatchValues | float,
    ) -> ActivationsVector:
        """
        Calculate the direction for a group of activations.

        Parameters
        ----------
        activations_group: BatchActivationsVector
            Activations from the group
        mean: ActivationsVector
            The mean of all activations
        sign: BatchValues | float
            The sign of the direction.
            -1 for the "from" group and 1 for the "to" group.
            Can be either a float or a tensor of batch size.

        Returns
        -------
        ActivationsVector
            The direction vector
        """
        if isinstance(sign, pt.Tensor):
            sign = sign.unsqueeze(1)
        return pt.mean(
            sign * (activations_group - mean),
            dim=0,
        )

    @property
    def classifying_direction(self) -> ActivationsVector:
        """
        Calculate the classifying direction, the vector that points from one group to
        the other.

        Note that this has a magnitude which is half distance between the two groups
        (in the balanced case).

        If balance is True, weights the two groups equally, otherwise it will be biased
        by the number of datapoints in each group.

        Returns
        -------
        ActivationsVector
            The classifying direction vector
        """
        if self.balance:
            return 0.5 * (
                self._calculate_direction_for_group(
                    self.activations_to, mean=self.mean_activations, sign=1
                )
                + self._calculate_direction_for_group(
                    self.activations_from, mean=self.mean_activations, sign=-1
                )
            )

        sign = pt.cat(
            [
                -1 * pt.ones(self.activations_from.shape[0]),
                pt.ones(self.activations_to.shape[0]),
            ]
        )
        return self._calculate_direction_for_group(
            pt.cat([self.activations_from, self.activations_to], dim=0),
            mean=self.mean_activations,
            sign=sign,
        )

    def get_distance_along_classifying_direction(
        self,
        batch_activations: BatchActivationsVector,
        center_from_origin: bool = False,
    ) -> BatchValues:
        """
        Calculate the distance of each activation along the classifying direction.
        This is the magnitude of the projection onto the normalized classifying
        direction.

        Parameters
        ----------
        batch_activations: BatchActivationsVector
            The activations to calculate the distance along the classifying direction
        center_from_origin: bool, default=False
            Whether to calculate the distance from the origin (mean of all activations)
            or from the centroid of the group.

        Returns
        -------
        BatchValues
            The signed magnitude of the projection along the classifying direction
        """
        direction = self.classifying_direction
        norm = pt.norm(direction)
        if norm < 1e-8:
            normalized_direction = pt.zeros_like(direction)
        else:
            normalized_direction = direction / norm

        if not center_from_origin:
            batch_activations = batch_activations - self.mean_activations
        return (batch_activations) @ normalized_direction
