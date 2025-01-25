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
        self, activations: BatchActivationsVector, labels: np.typing.ArrayLike
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
        self.activations = activations
        self.labels = pd.Series(labels).copy().reset_index(drop=True)

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

        groups_handlers = [self.get_groups(labels=group) for group in labels]
        return self.__class__(
            activations=pt.cat(
                [
                    group_handler.activations
                    for group_handler in groups_handlers
                ],
                dim=0,
            ),
            labels=pd.concat(
                [group_handler.labels for group_handler in groups_handlers]
            ),
        )

    def shuffle(self) -> ActivationsHandler:
        """
        Shuffle the dataset.

        Returns
        -------
        ActivationsHandler
            A shuffled ActivationsHandler
        """
        indices = list(self.labels.sample(frac=1, replace=False).index)
        return self.__class__(
            activations=self.activations[indices],
            labels=self.labels[indices],
        )

    def split_dataset(
        self, split_sizes: list | tuple, random: bool = True
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
        random: bool = True,
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
                group_labels=group_labels, random=False
            )

        group_handlers = [
            self.get_groups(group_label) for group_label in group_labels
        ]
        n_per_group = min(
            len(group_handler.labels) for group_handler in group_handlers
        )
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
