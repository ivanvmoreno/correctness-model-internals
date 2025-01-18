from typing import Any

import torch as pt


class Activations:
    def __init__(self, activations: pt.Tensor, labels: Any):
        self.activations = activations
        self.labels = labels

        self.groups = {
            label: self.activations[self.labels[self.labels == label]]
            for label in self.labels
        }

    def get_activations(
        self, subset: None | list | set | tuple | Any = None
    ) -> pt.Tensor:
        if subset is None:
            return self.activations

        if isinstance(subset, (list, set, tuple, pt.Tensor)):
            return pt.cat([self.groups[label] for label in subset], axis=0)
        return self.groups[subset]
