from typing import Any

import numpy as np
import pandas as pd
import torch as pt


class ActivationsHandler:
    def __init__(self, activations: pt.Tensor, labels: Any):
        self.activations = activations
        self.labels = pd.Series(labels).reset_index(drop=True)

        # self.groups = {
        #     label: list(self.labels[self.labels == label].index)
        #     for label in self.labels
        # }

    def get_activation_groups(
        self, groups: None | list | set | tuple | Any = None
    ) -> pt.Tensor:
        if groups is None:
            return self.activations

        if not isinstance(groups, (list, set, tuple, pt.Tensor)):
            if groups not in self.labels.unique():
                raise ValueError(f"{groups} is not a valid label")
            labels = self.labels[self.labels == groups]
            return ActivationsHandler(
                activations=self.activations[labels.index],
                labels=labels.reset_index(drop=True),
            )

        groups_handlers = [
            self.get_activation_groups(groups=group) for group in groups
        ]
        return ActivationsHandler(
            activations=pt.cat(
                [
                    group_handler.activations
                    for group_handler in groups_handlers
                ],
                axis=0,
            ),
            labels=pd.concat(
                [group_handler.labels for group_handler in groups_handlers]
            ).reset_index(drop=True),
        )

    def split_dataset(self, splits: list | tuple):
        n_entries = len(self.labels)

        splits = np.array(splits)
        splits = splits / np.sum(splits)
        loc_splits = [0] + list(
            int(x) for x in n_entries * np.cumsum(splits)
        )  # get the splits as row numbers

        indices = list(range(n_entries))
        np.random.shuffle(indices)

        for prev_loc_split, next_loc_split in zip(
            loc_splits[:-1], loc_splits[1:]
        ):
            indices_ = indices[prev_loc_split:next_loc_split]
            yield ActivationsHandler(
                activations=self.activations[indices_],
                labels=self.labels[indices_],
            )

    def sample_equally_across_groups(self, group_labels: list | set | tuple):
        labels = pd.Series(self.labels).reset_index(
            drop=True
        )  # index of series aligns with activations
        labels = labels[labels.isin(group_labels)]

        group_handlers = [
            self.get_activation_groups(group_label)
            for group_label in group_labels
        ]
        n_per_group = min(
            len(group_handler.labels) for group_handler in group_handlers
        )
        return ActivationsHandler(
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
