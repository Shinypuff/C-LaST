import torch


def weighted_smoothing_spline_1d(
    x: torch.Tensor,
    y: torch.Tensor,
    knots: torch.Tensor,
    weights: torch.Tensor,
    smoothing=1.0,
):
    """Weighted smoothing spline interpolation in PyTorch.

    Args:
        x: 1D tensor of input x-coordinates (must be sorted)
        y: 1D tensor of input y-coordinates
        knots: 1D tensor of knots (unique values from x).
        weights: weights for each data point
        smoothing: Controls trade-off between fit and smoothness

    Returns:
        A function that computes the spline at new x values

    source: https://grodri.github.io/demography/smoothing.pdf

    """
    n = len(knots)

    # Basis functions
    def design_matrix(x_vals: torch.Tensor, knots: torch.Tensor):
        """Calculate the values of the basis functions at x_vals, given knots.

        Returns a tensor of shape (n_values, n_knots + 4) where n is the number of knots.
        """
        x_vals_ = x_vals.unsqueeze(-1)
        knots_ = knots.unsqueeze(0)
        knots_d1 = knots_.diff(dim=1, n=1)
        knots_d2 = knots_.diff(dim=1, n=2)
        knots_d3 = knots_.diff(dim=1, n=3)
        B0 = (knots_[:-1] < x_vals_) & (x_vals_ < knots_[1:])
        B1 = x_vals_ - knots
        # X_poly = torch.stack(
        #     [torch.ones_like(x_vals), x_vals, x_vals**2, x_vals**3], dim=1
        # )
        # X_trunc = torch.relu(x_vals.unsqueeze(-1) - knots) ** 3
        # return torch.cat([X_poly, X_trunc], dim=1).nan_to_num(0, 0, 0)

    X = design_matrix(x, knots)
    xknots = design_matrix(knots, knots)

    def roughness_penalty(knots: torch.Tensor):
        fin_mask = knots.isfinite()
        h = torch.diff(knots).nan_to_num(1)
        r = h.reciprocal()
        D = knots.new_zeros(n - 2, n)
        D[:, :-2] += torch.diag(r[:-1])
        D[:, 1:-1] += torch.diag(-r[:-1] - r[1:])
        D[:, 2:] += torch.diag(r[1:])

        Vl = torch.diag(h[:-2] / 6, 1)
        Vd = torch.diag((h[:-1] + h[1:]) / 3)
        V = Vl + Vd + Vl.T

        # The penalty matrix K should be of size (n+4, n+4) since we have:
        # - 4 polynomial terms (1, x, x², x³)
        # - n truncated power terms
        # We only want to penalize the truncated power terms, not the polynomial terms
        K_full = knots.new_zeros(n + 4, n + 4)
        K_full[4:, 4:] = D.T @ torch.linalg.solve(
            V, D
        )  # Only apply penalty to truncated terms

        # Remove padding penalty
        K_full[4:, 4:] = torch.where(
            fin_mask[None, :] & fin_mask[:, None], K_full[4:, 4:], 0
        )
        return K_full

    K = roughness_penalty(knots)

    # Solve weighted regularized least squares
    W = torch.diag(weights)
    A = X.T @ W @ X + smoothing * K
    A = A + torch.eye(A.shape[0], device=x.device, dtype=x.dtype) * 1e-9
    b = X.T @ W @ y

    coeffs = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
    return xknots @ coeffs
