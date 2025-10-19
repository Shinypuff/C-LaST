from torch import Tensor

from ..maths.gaussian import gaussian_kldiv


def gaussian_kldiv_loss(
    mu1: Tensor, sigma1: Tensor, mu2: Tensor | None = None, sigma2: Tensor | None = None
):
    if mu2 is None:
        mu2 = mu1.new_tensor(0)

    if sigma2 is None:
        sigma2 = sigma1.new_tensor(1)

    kldiv = gaussian_kldiv(mu1, sigma1, mu2, sigma2)

    return kldiv.mean()
