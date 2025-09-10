"""File with the Gaussian Negative Log Likelihood loss."""

from torch import Tensor

from ..maths.gaussian import gaussian_log_pdf


def gaussian_nll(y_true: Tensor, mean: Tensor, scale: Tensor):
    """Compute the Gaussian Negative Log-Likelihood (NLL) loss.

    This function calculates the NLL loss assuming a Gaussian distribution parameterized by mean and scale.
    It standardizes the true values using the mean and scale, then computes the log probability density
    and returns the negative mean of the log probabilities adjusted by the scale.

    Args:
        y_true (Tensor): The true target values.
        mean (Tensor): The predicted mean values.
        scale (Tensor): The predicted scale (standard deviation) values.

    Returns:
        Tensor: The computed Gaussian NLL loss.

    """
    y0 = (y_true - mean) / scale
    loss = -(gaussian_log_pdf(y0) - scale).mean()
    return loss
