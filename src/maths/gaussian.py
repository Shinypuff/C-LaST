"""File with some utilities for the Gaussian distribution."""

import math

import torch
from torch import Tensor


def gaussian_cdf(x: Tensor):
    """Gaussian cumulative density function."""
    return (1 + torch.erf(x / math.sqrt(2))) / 2


def gaussian_pdf(x: Tensor):
    """Gaussian probability density function."""
    return torch.exp(-(x**2) / 2) / math.sqrt(2 * math.pi)


def gaussian_log_pdf(x: Tensor):
    """Gaussian log probability density function."""
    return -(x**2) / 2 - math.log(2 * math.pi) / 2
