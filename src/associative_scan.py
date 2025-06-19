import torch
from torch import Tensor

from .mask_utils import maskroll


def complex_log(float_input, eps=1e-6):
    """Compute the complex logarithm.

    Used in associative_scan.
    """
    eps = float_input.new_tensor(eps)
    real = float_input.abs().maximum(eps).log()
    imag = (float_input < 0).to(float_input.dtype) * torch.pi
    return torch.complex(real, imag)


def associative_scan(values: Tensor, coeffs: Tensor, dim: int):
    """Calculate cumsum with resets.

    Source: https://github.com/pytorch/pytorch/issues/53095#issuecomment-2102409471.

    Examples
    --------
    >>> input = torch.tensor([1, 2, 3, 4, 5])
    >>> inverted_reset_mask = torch.tensor([0, 1, 1, 0, 1])
    >>> output = associative_scan(input, inverted_reset_mask, dim=0)
    >>> print(output)
    tensor([1.0000, 3.0000, 6.0000, 4.0000, 9.0000])

    """
    log_values = complex_log(values.float())
    log_coeffs = complex_log(coeffs.float())
    a_star = torch.cumsum(log_coeffs, dim=dim)
    log_x0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=dim)
    log_x = a_star + log_x0_plus_b_star
    return torch.exp(log_x).real


def average_simultaneous(
    x: torch.Tensor, w: torch.Tensor, t: torch.Tensor, m: torch.Tensor
):
    """Compute the average of values x weighted by w for simultaneous observations.

    This function processes simultaneous observations along a tensor dimension by performing an associative scan over
    the input values to accumulate weighted averages. It identifies new and repeating timestamps, calculates the cumulative
    weighted sum and weight of simultaneous values, then averages them while preserving non-simultaneous observations.

    Args:
        x (torch.Tensor): Tensor of values to average.
        w (torch.Tensor): Tensor of corresponding weights for values in x.
        t (torch.Tensor): Tensor of timestamps associated with x.
        m (torch.Tensor): Boolean mask tensor used to denote valid observations.

    Returns:
        tuple: A tuple containing:
            m_avg (torch.Tensor): Updated mask tensor for the averaged values.
            x_avg (torch.Tensor): Averaged values based on simultaneous observations.
            w_avg (torch.Tensor): Accumulated weights corresponding to the averaged values.

    Note:
        For each group of average observations x_i, they are replaced with sum(w_i x_i) / sum(w_i)

    """
    B, L = t.shape

    # Is the current observation new?
    is_new = t.new_empty(B, L, dtype=bool)
    is_new[:, 0] = True
    is_new[:, 1:] = t[:, :-1] != t[:, 1:]

    # Is the current observation repeating the previous one?
    is_repeating = ~is_new

    # Is the next observation new?
    next_new = torch.roll(is_new, -1, 1)

    while is_repeating.ndim < x.ndim:
        is_repeating = is_repeating.unsqueeze(-1)

    x_acc = associative_scan(x * w, is_repeating, dim=1)
    w_acc = associative_scan(w, is_repeating, dim=1)
    x_avg = x_acc / w_acc

    # keep last in every simultaneous group
    m_avg, x_avg, w_avg = maskroll(m & next_new, x_avg, w_acc)
    return x_avg, w_avg, m_avg
