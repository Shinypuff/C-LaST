"""The file with the Normalized Mean Absolute Error (NMAE) metric."""

import torch
from torch import Tensor
from torchmetrics import Metric


class NMAE(Metric):
    """Normalized Mean Absolute Error (NMAE) metric."""

    def __init__(self, **kwargs):
        """Initialize the NMAE metric."""
        super().__init__(**kwargs)
        self.add_state(
            "absolute_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("y_true", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """Update the NMAE metric: add the absolute error and the true sum to internal state.

        Args:
            preds: The predictions.
            target: The targets.

        """
        self.absolute_error += (preds - target).abs().sum()
        self.y_true += target.abs().sum()

    def compute(self):
        """Compute the NMAE metric, dividing the absolute error by the true sum.

        Returns:
            The NMAE metric.

        """
        return self.absolute_error / self.y_true
