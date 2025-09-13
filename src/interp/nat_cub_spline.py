"""The module with natural cubic interpolation functions."""

import torch
from torch import Tensor, nn


def tdmasolver(a: Tensor, b: Tensor, c: Tensor, d: Tensor):
    """Solve Tri-Diagonal system of equations.

    This is a vmap-compatible version of https://gist.github.com/TheoChristiaanse/d168b7e57dd30342a81aa1dc4eb3e469
    Refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    """
    nf = d.shape[0]  # number of equations

    ac, bc, cc, dc = (
        list(torch.unbind(a.clone())),
        list(torch.unbind(b.clone())),
        list(torch.unbind(c.clone())),
        list(torch.unbind(d.clone())),
    )  # Unbind first dimension

    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return torch.stack(xc)


def fit_cubic_spline_1d(t: Tensor, x: Tensor):
    """Fit a 1-dimensional cubic spline.

    This function is a vmap-able version of the torchcde interpolation:
    https://github.com/patrick-kidger/torchcde/blob/9ff6aba4738989dc5fe3aee86d45812c318f6231/torchcde/interpolation_cubic.py#L7

    Args:
    ----
        t (Tensor): shape (L,) -- observation times;
        x (Tensor): shape (L,) -- observations.


    """
    # Handle infinite ts, which we introduced during preprocessing
    # We don't want to touch them since we will never reach them,
    # this seems to do the trick (otherwise tdmasolver yields nans).
    time_diffs = torch.where(t[1:].isfinite(), t.diff(dim=0), float("inf"))
    time_diffs_reciprocal = time_diffs.reciprocal()
    time_diffs_reciprocal_squared = time_diffs_reciprocal**2
    three_path_diffs = 3 * x.diff(dim=0)
    six_path_diffs = 2 * three_path_diffs
    path_diffs_scaled = three_path_diffs * time_diffs_reciprocal_squared

    # Solve a tridiagonal linear system to find the derivatives at the knots
    system_diagonal = torch.full_like(x, fill_value=1e-9)
    system_diagonal[:-1] += time_diffs_reciprocal
    system_diagonal[1:] += time_diffs_reciprocal
    system_diagonal *= 2
    system_rhs = torch.zeros_like(x)
    system_rhs[:-1] = path_diffs_scaled
    system_rhs[-1] = 0
    system_rhs[1:] += path_diffs_scaled

    knot_derivatives = tdmasolver(
        time_diffs_reciprocal, system_diagonal, time_diffs_reciprocal, system_rhs
    )

    # Do some algebra to find the coefficients of the spline
    a = x[:-1]
    b = knot_derivatives[:-1]

    two_c = (
        six_path_diffs * time_diffs_reciprocal
        - 4 * knot_derivatives[:-1]
        - 2 * knot_derivatives[1:]
    ) * time_diffs_reciprocal
    three_d = (
        -six_path_diffs * time_diffs_reciprocal
        + 3 * (knot_derivatives[:-1] + knot_derivatives[1:])
    ) * time_diffs_reciprocal_squared
    return torch.stack([a, b, two_c / 2, three_d / 3])


def eval_cubic_spline_1d(coeffs: Tensor, t_obs: Tensor, t_eval: Tensor):
    """Interpolate at t_eval.

    Args:
    ----
        coeffs: 4-tuple of Tensors of shape (4, L - 1, *)
        t_obs: Tensor of shape (L,)
        t_eval: single-item tensor.

    Returns:
    -------
        A tensor of shape (*,), corresponding to the value of the spline
        at `t_eval`.

    """
    # (B, C), i.e. need to unsqueeze(1) to gather t_obs along L.
    idx = torch.clamp(torch.searchsorted(t_obs, t_eval) - 1, 0, coeffs[0].size(0) - 1)
    selected_obs = t_obs.index_select(0, idx.unsqueeze(0))
    selected_coeffs = coeffs.index_select(1, idx.unsqueeze(0))[:, 0]
    rem = t_eval - selected_obs
    rem_pow = rem ** torch.arange(1, 4, device=rem.device)  # (4,)
    res = rem_pow @ selected_coeffs[1:] + selected_coeffs[0]
    return res


fit_cubic_spline = torch.vmap(torch.vmap(fit_cubic_spline_1d), (None, -1), -1)
eval_cubic_spline = torch.vmap(torch.vmap(eval_cubic_spline_1d), (-1, None, None), -1)
