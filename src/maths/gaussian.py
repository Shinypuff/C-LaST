import math

import torch
from torch import Tensor


def gaussian_cdf(x: Tensor):
    return (1 + torch.erf(x / math.sqrt(2))) / 2


def gaussian_pdf(x: Tensor):
    return torch.exp(-(x**2) / 2) / math.sqrt(2 * math.pi)


def gaussian_log_pdf(x: Tensor):
    return -(x**2) / 2 - math.log(2 * math.pi) / 2
