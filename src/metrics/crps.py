"""File with the CRPS metric.

A large part was taken from torchmetrics, but this is a more memory-efficient implementation.
"""

from typing import Any, Tuple

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_same_shape


def _crps_update(preds: Tensor, target: Tensor) -> Tuple[int, Tensor, Tensor]:
    """Compute intermediate CRPS values before aggregation.

    Args:
        preds: Tensor of shape (batch_size, ensemble_members)
        target: Tensor of shape (batch_size,)

    Returns:
        batch_size: int
        diff: Tensor (batch-wise absolute error term)
        ensemble_sum: Tensor (pairwise ensemble term)

    """
    # Only second dimension should deviate in shape (the ensemble members)
    _check_same_shape(preds[:, 0], target)
    batch_size, n_ensemble_members = preds.shape
    if n_ensemble_members < 2:
        raise ValueError(
            f"CRPS requires at least 2 ensemble members, but you provided {preds.shape}."
        )

    # sort forecasts
    preds = torch.sort(preds).values

    # inflate observations:
    observation_inflated = target.unsqueeze(1).expand_as(preds)

    # Compute mean absolute difference between predictions and target
    diff = (
        torch.sum(torch.abs(preds - observation_inflated), dim=1) / n_ensemble_members
    )

    # Compute ensemble term using the mean-of-abs-diff formula
    ### CHANGED FROM HERE ###
    shifts_arange = torch.arange(
        -(n_ensemble_members - 1), n_ensemble_members, 2, dtype=preds.dtype
    )  # [-(n-1), n-1, 2]

    ensemble_sum = (preds @ shifts_arange) / n_ensemble_members**2
    ### END OF CHANGE ###

    return batch_size, diff, ensemble_sum


class ContinuousRankedProbabilityScore(Metric):
    r"""Computes continuous ranked probability score.

    .. math::
        CRPS(F, y) = \int_{-\infty}^{\infty} (F(x) - 1_{x \geq y})^2 dx

    where :math:`F` is the predicted cumulative distribution function and :math:`y` is the true target. The metric is
    usually used to evaluate probabilistic regression models, such as forecasting models. A lower CRPS indicates a
    better forecast, meaning that forecasted probabilities are closer to the true observed values. CRPS can also be
    seen as a generalization of the brier score for non binary classification problems.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): Predicted float tensor with shape ``(N,d)``
    - ``target`` (:class:`~torch.Tensor`): Ground truth float tensor with shape ``(N,d)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``cosine_similarity`` (:class:`~torch.Tensor`): A float tensor with the cosine similarity

    Args:
        reduction: how to reduce over the batch dimension using 'sum', 'mean' or 'none' (taking the individual scores)
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torch import randn
        >>> from torchmetrics.regression import ContinuousRankedProbabilityScore
        >>> preds = randn(10, 5)
        >>> target = randn(10)
        >>> crps = ContinuousRankedProbabilityScore()
        >>> crps(preds, target)
        tensor(0.7731)

    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    score: Tensor
    total: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("score", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values

        """
        batch_size, diff, ensemble_sum = _crps_update(preds, target)
        self.score += torch.sum(diff - ensemble_sum)
        self.total += batch_size

    def compute(self) -> Tensor:
        """Compute the continuous ranked probability score over state."""
        return self.score / self.total
