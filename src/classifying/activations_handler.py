from __future__ import annotations

from typing import Any, Generator

import numpy as np
import pandas as pd
import torch as pt

from .typing import BatchActivationsVector


class ActivationsHandler:
    """
    Utility class for handling activations and labels.

    Attributes
    ----------
    activations: BatchActivationsVector
        Activations, rows are entries (datapoints), columns are activations from nodes
        from the MLP
    labels: pd.Series
        Labels (i.e. True/False or anything else), the index of the series should
        correspond to the row in activations.
    """

    def __init__(
        self,
        activations: BatchActivationsVector,
        labels: np.typing.ArrayLike,
        _allow_empty: bool = False,
    ):
        """
        Setup

        Parameters
        ----------
        activations: BatchActivationsVector
            The activations to set
        labels: np.typing.ArrayLike
            Must align with activations
        """
        if activations.shape[0] != len(labels):
            raise ValueError("activations and labels must have the same length")
        if len(labels) == 0 and not _allow_empty:
            raise ValueError("labels and activations must not be empty")

        self.activations = activations
        self.labels = pd.Series(labels).copy().reset_index(drop=True)
        self.count = len(self.labels)

    def __add__(self, other: ActivationsHandler) -> ActivationsHandler:
        """
        Add two ActivationsHandlers together.

        Parameters
        ----------
        other: ActivationsHandler
            The other ActivationsHandler to add

        Returns
        -------
        ActivationsHandler
            The combined ActivationsHandler
        """
        if other.labels.empty:
            raise ValueError("other.labels is empty")

        if self.labels.empty:
            activations = other.activations
            labels = other.labels
        elif other.labels.empty:
            activations = self.activations
            labels = self.labels
        else:
            activations = pt.cat([self.activations, other.activations], dim=0)
            labels = pd.concat([self.labels, other.labels])

        return self.__class__(
            activations=activations,
            labels=labels,
        )

    def get_groups(
        self, labels: list | set | tuple | Any
    ) -> ActivationsHandler:
        """
        Get a new ActivationsHandler instance depending on the label(s)

        Parameters
        ----------
        labels: list | set | tuple | Any
            The label (or group of labels) to create ActivationsHandlers for.

        Returns
        -------
        ActivationsHandler
            A single ActivationsHandler with only the defined activations and
            labels for the given split(s)
        """
        if not isinstance(labels, (list, set, tuple)):
            # if just a single label
            if labels not in self.labels.unique():
                raise ValueError(f"{labels} is not a valid label")
            labels_ = self.labels[self.labels == labels]
            return self.__class__(
                activations=self.activations[labels_.index],
                labels=labels_,
            )

        return combine_activations_handlers(
            [self.get_groups(labels=group) for group in labels]
        )

    def sample(self, frac: float, random: bool = False) -> ActivationsHandler:
        """
        Sample a fraction of the dataset.

        Parameters
        ----------
        frac: float
            The fraction of the dataset to sample
        random: bool
            Whether to shuffle the dataset before sampling

        Returns
        -------
        ActivationsHandler
            A sampled ActivationsHandler
        """
        if random:
            indices = list(self.labels.sample(frac=frac, replace=False).index)
        else:
            indices = list(self.labels.iloc[: int(self.count * frac)].index)
        return self.__class__(
            activations=self.activations[indices],
            labels=self.labels[indices],
        )

    def shuffle(self) -> ActivationsHandler:
        """
        Shuffle the dataset.

        Returns
        -------
        ActivationsHandler
            A shuffled ActivationsHandler
        """
        return self.sample(frac=1.0, random=True)

    def split_dataset(
        self, split_sizes: list | tuple, random: bool = False
    ) -> Generator[ActivationsHandler, None, None]:
        """
        A generator of ActivationsHandlers depending on the split.

        Parameters
        ----------
        split_sizes: list | tuple
            To define the new set sizes.
            The sum is normalised so you can do for example:
            - [0.8, 0.2] for an 80, 20 split
            - [50, 50] for a 50/50 split,
            - [1, 2, 1, 6] for a split of 10%, 20%, 10%, 60%
        random: bool
            Whether to shuffle the dataset before splitting

        Returns
        -------
        Generator[ActivationsHandler, None, None]
            Yields an ActivationsHandler for each of the split sizes
        """
        if len(split_sizes) < 1:
            raise ValueError("Must define more than one split_sizes")

        if random:
            yield from self.shuffle().split_dataset(
                split_sizes=split_sizes, random=False
            )
            return

        n_entries = len(self.labels)

        split_sizes = np.array(split_sizes)
        if np.any(split_sizes <= 0):
            raise ValueError("Splits must be positive numbers.")

        split_sizes = split_sizes / np.sum(split_sizes)

        loc_splits = [0] + list(
            np.round(n_entries * np.cumsum(split_sizes)).astype(int)
        )

        indices = list(range(n_entries))

        for prev_loc_split, next_loc_split in zip(
            loc_splits[:-1], loc_splits[1:]
        ):
            indices_ = indices[prev_loc_split:next_loc_split]
            yield self.__class__(
                activations=self.activations[indices_],
                labels=self.labels.iloc[indices_].reset_index(drop=True),
            )

    def sample_equally_across_groups(
        self,
        group_labels: list | set | tuple,
        random: bool = False,
        interleave: bool = False,
    ) -> ActivationsHandler:
        """
        Get an ActivationsHandler with an equal number of samples from each group.

        Parameters
        ----------
        group_labels: list | set | tuple
            The labels to sample equally from.
        random: bool
            Whether to shuffle the dataset before splitting

        Returns
        -------
        ActivationsHandler
            An ActivationsHandler with an equal number of samples from each group
        """
        if random:
            return self.shuffle().sample_equally_across_groups(
                group_labels=group_labels, random=False, interleave=interleave
            )

        group_handlers = [
            self.get_groups(group_label) for group_label in group_labels
        ]
        n_per_group = min(
            len(group_handler.labels) for group_handler in group_handlers
        )
        if not interleave:
            return self.__class__(
                activations=pt.cat(
                    [
                        group_handler.activations[:n_per_group]
                        for group_handler in group_handlers
                    ]
                ),
                labels=pd.concat(
                    [
                        group_handler.labels.iloc[:n_per_group]
                        for group_handler in group_handlers
                    ]
                ),
            )

        activations = pt.zeros(
            (
                len(group_handlers) * n_per_group,
                group_handlers[0].activations.shape[-1],
            )
        )
        labels = pd.Series(
            dtype=bool, index=range(len(group_handlers) * n_per_group)
        )

        for i in range(n_per_group):
            for j, group_handler in enumerate(group_handlers):
                activations[i * len(group_handlers) + j, :] = (
                    group_handler.activations[i, :]
                )
                labels.iloc[i * len(group_handlers) + j] = (
                    group_handler.labels.iloc[i]
                )
        return self.__class__(activations=activations, labels=labels)

    def reduce_dims(
        self,
        pca_components: int | None,
        pca_info: tuple[pt.Tensor, pt.Tensor] | None = None,
    ) -> tuple[ActivationsHandler, pt.Tensor]:
        """
        Reduce the dimensionality of the activations by taking the top PCA components.

        Parameters
        ----------
        pca_components: int | None
            The number of components to reduce the dimensionality to

        Returns
        -------
        ActivationsHandler
            The reduced ActivationsHandler
        tuple[pt.Tensor, pt.Tensor]
            The Vh matrix from the SVD, and the mean of the activations
            used to center the data.
            Use this for example if the PCA had been done on another
            train set and this is the test set.
        """
        if pca_components is None:
            return self, None

        if pca_info is None:
            # Center the data
            mean_activations = self.activations.mean(dim=0)
            activations = self.activations - mean_activations
            # Perform SVD directly on centered data
            U, S, Vh = pt.linalg.svd(activations, full_matrices=False)
            # Take top components
        else:
            Vh, mean_activations = pca_info
            activations = self.activations - mean_activations

        return (
            self.__class__(
                activations=activations @ Vh.T[:, :pca_components],
                labels=self.labels,
            ),
            (Vh, mean_activations),
        )


def combine_activations_handlers(
    activations_handlers: list[ActivationsHandler], equal_counts: bool = False
) -> ActivationsHandler:
    """
    Combine a list of ActivationsHandlers into a single ActivationsHandler.

    Parameters
    ----------
    activations_handlers: list[ActivationsHandler]
        The list of ActivationsHandlers to combine

    Returns
    -------
    ActivationsHandler
        The combined ActivationsHandler
    """
    if len(activations_handlers) == 1:
        return activations_handlers[0]

    if equal_counts:
        min_len = min(len(ah.labels) for ah in activations_handlers)
        activations_handlers = [
            ah.sample(min_len / len(ah.labels)) for ah in activations_handlers
        ]
    return sum(
        activations_handlers,
        start=ActivationsHandler(
            activations=pt.tensor([]), labels=pd.Series([]), _allow_empty=True
        ),
    )
