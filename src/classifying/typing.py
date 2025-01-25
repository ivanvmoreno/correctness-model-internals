from typing import Annotated

import torch as pt

# Each row is a feature vector
BatchActivationsVector = Annotated[pt.Tensor, "batch", "activations"]
# Single feature vector
ActivationsVector = Annotated[pt.Tensor, "activations"]
# Single value per batch item
BatchValues = Annotated[pt.Tensor, "batch"]
