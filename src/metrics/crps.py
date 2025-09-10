"""File with the CRPS metric."""

import math

import torch
from torch import Tensor
from torchmetrics import Metric

from ..maths.gaussian import gaussian_cdf, gaussian_pdf


def calc_crps(y_true: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
    """Calculate the CRPS metric, assuming normally-distributed predictions.

    Args:
        y_true: The true values of the target variable.
        mean: The predicted mean of the target variable.
        scale: The predicted scale of the target variable.

    Returns:
        The CRPS metric.

    Note:
        The formula is taken from Gneitig et al., 2005:
        https://journals.ametsoc.org/downloadpdf/view/journals/mwre/133/5/mwr2904.1.pdf


    """
    y0 = (y_true - mean) / scale
    cdf = gaussian_cdf(y0)
    pdf = gaussian_pdf(y0)

    # When scale -> 0, CRPS tends to MAE.
    return torch.where(
        scale == 0,
        (y_true - mean).abs(),
        scale * (y0 * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(torch.pi)),
    )


class CRPS(Metric):
    """Calculate the CRPS metric."""

    def __init__(self, **kwargs):
        """Initialize the CRPS metric."""
        super().__init__(**kwargs)
        self.add_state("crps", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_true: Tensor, mean, scale):
        """Calculate the crps metric on new predictions, add it to internal state.

        Args:
            y_true: The true values of the target variable.
            mean: The predicted mean of the target variable.
            scale: The predicted scale of the target variable.

        """
        crps = calc_crps(y_true, mean, scale)
        self.crps += crps.sum()
        self.n += y_true.numel()

    def compute(self):
        """Compute the CRPS metric."""
        return self.crps / self.n
