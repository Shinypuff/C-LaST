from torch import Tensor

from ..maths.gaussian import gaussian_log_pdf


def gaussian_nll(y_true: Tensor, mean: Tensor, scale: Tensor):
    y0 = (y_true - mean) / scale
    loss = -(gaussian_log_pdf(y0) - scale).mean()
    return loss
