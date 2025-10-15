"""Standalone version of Structured (Sequence) State Space (S4) model."""

import logging
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
from einops import rearrange, repeat
import opt_einsum as oe

contract = oe.contract
contract_expression = oe.contract_expression


def power(L, A, v=None):
    """Compute A^L and the scan sum_i A^i v_i

    A: (..., N, N)
    v: (..., N, L)
    """

    I = torch.eye(A.shape[-1]).to(A)  # , dtype=A.dtype, device=A.device)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1:
            I = powers[-1] @ I
        L //= 2
        if L == 0:
            break
        l *= 2
        powers.append(powers[-1] @ powers[-1])

    if v is None:
        return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, "... (z l) -> ... z l", z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)


def transition(measure, N):
    """A, B transition matrices for different measures"""
    # Legendre (translated)
    if measure == "legt":
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1) ** 0.5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.0) ** (i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A

        # Halve again for timescale correctness
        A *= 0.5
        B *= 0.5
    # Legendre (scaled)
    elif measure == "legs":
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = (
            B.copy()
        )  # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == "legsd":
        # Essentially equivalent to S4D-LegS
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = (
            B.copy()
        )  # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
        A += 0.5 * B * B[None, :, 0]
        B = B / 2.0
    elif measure in ["fourier_diag", "foud"]:
        # Essentially equivalent to S4D-Lin
        freqs = np.arange(N // 2)
        d = np.stack([freqs, np.zeros(N // 2)], axis=-1).reshape(-1)[:-1]
        A = 2 * np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        A = A - 0.5 * np.eye(N)
        B = np.zeros(N)
        B[0::2] = 2**0.5
        B[0] = 1
        B = B[:, None]
    elif measure in ["fourier", "fout"]:
        freqs = np.arange(N // 2)
        d = np.stack([np.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
        A = np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2**0.5
        B[0] = 1

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - B[:, None] * B[None, :]
        B = B[:, None]
    else:
        raise NotImplementedError

    return A, B


def rank_correction(measure, N, rank=1, dtype=torch.float):
    """Return low-rank matrix L such that A + L is normal"""

    if measure == "legs":
        assert rank >= 1
        P = torch.sqrt(0.5 + torch.arange(N, dtype=dtype)).unsqueeze(
            0
        )  # (1 N)
    elif measure == "legt":
        assert rank >= 2
        P = torch.sqrt(1 + 2 * torch.arange(N, dtype=dtype))  # (N)
        P0 = P.clone()
        P0[0::2] = 0.0
        P1 = P.clone()
        P1[1::2] = 0.0
        P = torch.stack([P0, P1], dim=0)  # (2 N)
        P *= 2 ** (
            -0.5
        )  # Halve the rank correct just like the original matrix was halved
    elif measure in ["fourier", "fout"]:
        P = torch.zeros(N)
        P[0::2] = 2**0.5
        P[0] = 1
        P = P.unsqueeze(0)
    elif measure in ["fourier_diag", "foud", "legsd"]:
        P = torch.zeros(1, N, dtype=dtype)
    else:
        raise NotImplementedError

    d = P.size(0)
    if rank > d:
        P = torch.cat(
            [P, torch.zeros(rank - d, N, dtype=dtype)], dim=0
        )  # (rank N)
    return P


def nplr(measure, N, rank=1, dtype=torch.float, diagonalize_precision=True):
    """Return w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B
    """
    assert dtype == torch.float or dtype == torch.double
    cdtype = torch.cfloat if dtype == torch.float else torch.cdouble

    A, B = transition(measure, N)
    A = torch.as_tensor(A, dtype=dtype)  # (N, N)
    B = torch.as_tensor(B, dtype=dtype)[:, 0]  # (N,)

    P = rank_correction(measure, N, rank=rank, dtype=dtype)  # (r N)
    AP = A + torch.sum(P.unsqueeze(-2) * P.unsqueeze(-1), dim=-3)

    # We require AP to be nearly skew-symmetric
    _A = AP + AP.transpose(-1, -2)
    if (
        err := torch.sum((_A - _A[0, 0] * torch.eye(N)) ** 2) / N
    ) > 1e-5:  # if not torch.allclose(_A - _A[0,0]*torch.eye(N), torch.zeros(N, N), atol=1e-5):
        print("WARNING: HiPPO matrix not skew symmetric", err)

    # Take advantage of identity + skew-symmetric form to calculate real and imaginary parts separately
    # Imaginary part can use eigh instead of eig
    w_re = torch.mean(torch.diagonal(AP), -1, keepdim=True)

    # Diagonalize in double precision
    if diagonalize_precision:
        AP = AP.to(torch.double)
    w_im, V = torch.linalg.eigh(AP * -1j)  # (..., N) (..., N, N)
    if diagonalize_precision:
        w_im, V = w_im.to(cdtype), V.to(cdtype)
    w = w_re + 1j * w_im
    # Check: V w V^{-1} = A
    # print("check", V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2))

    # Only keep half of each conjugate pair
    _, idx = torch.sort(w.imag)
    w_sorted = w[idx]
    V_sorted = V[:, idx]

    # There is an edge case when eigenvalues can be 0, which requires some machinery to handle
    # We use a huge hack here: Assume only one pair is 0, and that it is the first row/column of A (only happens in Fourier case)
    V = V_sorted[:, : N // 2]
    w = w_sorted[: N // 2]
    assert (
        w[-2].abs() > 1e-4
    ), "Only 1 zero eigenvalue allowed in diagonal part of A"
    if w[-1].abs() < 1e-4:
        V[:, -1] = 0.0
        V[0, -1] = 2**-0.5
        V[1, -1] = 2**-0.5 * 1j

    _AP = V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2)
    if (err := torch.sum((2 * _AP.real - AP) ** 2) / N) > 1e-5:
        print(
            "Warning: Diagonalization of A matrix not numerically precise - error",
            err,
        )
    # print("check", V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2))

    V_inv = V.conj().transpose(-1, -2)

    B = contract("ij, j -> i", V_inv, B.to(V))  # V^* B
    P = contract("ij, ...j -> ...i", V_inv, P.to(V))  # V^* P

    return w, P, B, V


def dplr(
    scaling,
    N,
    rank=1,
    H=1,
    dtype=torch.float,
    real_scale=1.0,
    imag_scale=1.0,
    random_real=False,
    random_imag=False,
    normalize=False,
    diagonal=True,
    random_B=False,
):
    assert dtype == torch.float or dtype == torch.double
    dtype = torch.cfloat if dtype == torch.float else torch.cdouble

    pi = torch.tensor(math.pi)
    if random_real:
        real_part = torch.rand(H, N // 2)
    else:
        real_part = 0.5 * torch.ones(H, N // 2)
    if random_imag:
        imag_part = N // 2 * torch.rand(H, N // 2)
    else:
        imag_part = repeat(torch.arange(N // 2), "n -> h n", h=H)

    real_part = real_scale * real_part
    if scaling == "random":
        imag_part = torch.randn(H, N // 2)
    elif scaling == "real":
        imag_part = 0 * imag_part
        real_part = 1 + repeat(torch.arange(N // 2), "n -> h n", h=H)
    elif scaling in ["linear", "lin"]:
        imag_part = pi * imag_part
    elif scaling in [
        "inverse",
        "inv",
    ]:  # Based on asymptotics of the default HiPPO matrix
        imag_part = 1 / pi * N * (N / (1 + 2 * imag_part) - 1)
    elif scaling in ["inverse2", "inv2"]:
        imag_part = 1 / pi * N * (N / (1 + imag_part) - 1)
    elif scaling in ["quadratic", "quad"]:
        imag_part = 1 / pi * (1 + 2 * imag_part) ** 2
    elif scaling in ["legs", "hippo"]:
        w, _, _, _ = nplr("legsd", N)
        imag_part = w.imag

    else:
        raise NotImplementedError
    imag_part = imag_scale * imag_part
    w = -real_part + 1j * imag_part

    # Initialize B
    if random_B:
        B = torch.randn(H, N // 2, dtype=dtype)
    else:
        B = torch.ones(H, N // 2, dtype=dtype)

    if normalize:
        norm = (
            -B / w
        )  # (H, N) # Result if you integrate the kernel with constant 1 function
        zeta = 2 * torch.sum(
            torch.abs(norm) ** 2, dim=-1, keepdim=True
        )  # Variance with a random C vector
        B = B / zeta**0.5

    P = torch.randn(rank, H, N // 2, dtype=dtype)
    if diagonal:
        P = P * 0.0
    V = torch.eye(N, dtype=dtype)[:: N // 2]  # Only used in testing
    V = repeat(V, "n m -> h n m", h=H)

    return w, P, B, V


def ssm(measure, N, R, H, **ssm_args):
    """Dispatcher to create single SSM initialization

    N: state size
    R: rank (for DPLR parameterization)
    H: number of independent SSM copies
    """

    if measure == "dplr":
        w, P, B, V = dplr(N=N, rank=R, H=H, **ssm_args)
    elif measure.startswith("diag"):
        args = measure.split("-")
        assert args[0] == "diag" and len(args) > 1
        scaling = args[1]
        w, P, B, V = dplr(
            scaling=scaling, N=N, rank=R, H=H, diagonal=True, **ssm_args
        )
    else:
        w, P, B, V = nplr(measure, N, R, **ssm_args)
        w = repeat(w, "n -> s n", s=H)
        P = repeat(P, "r n -> r s n", s=H)
        B = repeat(B, "n -> s n", s=H)
        V = repeat(V, "n m -> s n m", s=H)
    return w, P, B, V


combinations = {
    "hippo": ["legs", "fourier"],
    "diag": ["diag-inv", "diag-lin"],
    "all": ["legs", "fourier", "diag-inv", "diag-lin"],
}


def combination(measures, N, R, S, **ssm_args):
    if isinstance(measures, str):
        measures = (
            combinations[measures] if measures in combinations else [measures]
        )

    assert (
        S % len(measures) == 0
    ), f"{S} independent trainable SSM copies must be multiple of {len(measures)} different measures"
    w, P, B, V = zip(
        *[
            ssm(measure, N, R, S // len(measures), **ssm_args)
            for measure in measures
        ]
    )
    w = torch.cat(w, dim=0)  # (S N)
    P = torch.cat(P, dim=1)  # (R S N)
    B = torch.cat(B, dim=0)  # (S N)
    V = torch.cat(V, dim=0)  # (S N N)
    return w, P, B, V
