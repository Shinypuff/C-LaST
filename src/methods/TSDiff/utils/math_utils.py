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

_c2r = torch.view_as_real
_r2c = torch.view_as_complex

""" Cauchy and Vandermonde kernels """

try:  # Try CUDA extension
    from extensions.cauchy.cauchy import cauchy_mult

    has_cauchy_extension = True
except ImportError:
    # log.warning(
    #     "CUDA extension for cauchy multiplication not found. Install by going to extensions/cauchy/ and running `python setup.py install`. This should speed up end-to-end training by 10-50%"
    # )
    has_cauchy_extension = False

try:  # Try pykeops
    from pykeops.torch import Genred

    has_pykeops = True
    # log.info("Pykeops installation found.")

    def _broadcast_dims(*tensors):
        max_dim = max([len(tensor.shape) for tensor in tensors])
        tensors = [
            tensor.view((1,) * (max_dim - len(tensor.shape)) + tensor.shape)
            for tensor in tensors
        ]
        return tensors

    def cauchy_conj(v, z, w):
        """Pykeops version"""
        expr_num = "z * ComplexReal(v) - Real2Complex(Sum(v * w))"
        expr_denom = "ComplexMult(z-w, z-Conj(w))"

        cauchy_mult = Genred(
            f"ComplexDivide({expr_num}, {expr_denom})",
            [
                "v = Vj(2)",
                "z = Vi(2)",
                "w = Vj(2)",
            ],
            reduction_op="Sum",
            axis=1,
        )

        v, z, w = _broadcast_dims(v, z, w)
        v = _c2r(v)
        z = _c2r(z)
        w = _c2r(w)

        r = 2 * cauchy_mult(v, z, w, backend="GPU")
        return _r2c(r)

    def log_vandermonde(v, x, L):
        expr = "ComplexMult(v, ComplexExp(ComplexMult(x, l)))"
        vandermonde_mult = Genred(
            expr,
            [
                "v = Vj(2)",
                "x = Vj(2)",
                "l = Vi(2)",
            ],
            reduction_op="Sum",
            axis=1,
        )

        l = torch.arange(L).to(x)
        v, x, l = _broadcast_dims(v, x, l)
        v = _c2r(v)
        x = _c2r(x)
        l = _c2r(l)

        r = vandermonde_mult(v, x, l, backend="GPU")
        return 2 * _r2c(r).real

    def log_vandermonde_transpose(u, v, x, L):
        """
        u: ... H L
        v: ... H N
        x: ... H N
        Returns: ... H N

        V = Vandermonde(a, L) : (H N L)
        contract_L(V * u * v)
        """
        expr = "ComplexMult(ComplexMult(v, u), ComplexExp(ComplexMult(x, l)))"
        vandermonde_mult = Genred(
            expr,
            [
                "u = Vj(2)",
                "v = Vi(2)",
                "x = Vi(2)",
                "l = Vj(2)",
            ],
            reduction_op="Sum",
            axis=1,
        )

        l = torch.arange(L).to(x)
        u, v, x, l = _broadcast_dims(u, v, x, l)
        u = _c2r(u)
        v = _c2r(v)
        x = _c2r(x)
        l = _c2r(l)

        r = vandermonde_mult(u, v, x, l, backend="GPU")
        return _r2c(r)

except ImportError:
    has_pykeops = False
    if not has_cauchy_extension:
        # log.warning(
        #     "Falling back on slow Cauchy kernel. Install at least one of pykeops or the CUDA extension for efficiency."
        # )

        def cauchy_naive(v, z, w):
            """
            v, w: (..., N)
            z: (..., L)
            returns: (..., L)
            """
            cauchy_matrix = v.unsqueeze(-1) / (
                z.unsqueeze(-2) - w.unsqueeze(-1)
            )  # (... N L)
            return torch.sum(cauchy_matrix, dim=-2)

    # Vandermonde functions
    # log.warning(
    #     "Falling back on slow Vandermonde kernel. Install pykeops for improved memory efficiency."
    # )

    def log_vandermonde(v, x, L):
        """
        v: (..., N)
        x: (..., N)
        returns: (..., L) \sum v x^l
        """
        vandermonde_matrix = torch.exp(
            x.unsqueeze(-1) * torch.arange(L).to(x)
        )  # (... N L)
        vandermonde_prod = contract(
            "... n, ... n l -> ... l", v, vandermonde_matrix
        )  # (... L)
        return 2 * vandermonde_prod.real

    def log_vandermonde_transpose(u, v, x, L):
        vandermonde_matrix = torch.exp(
            x.unsqueeze(-1) * torch.arange(L).to(x)
        )  # (... N L)
        vandermonde_prod = contract(
            "... l, ... n, ... n l -> ... n",
            u.to(x),
            v.to(x),
            vandermonde_matrix,
        )  # (... L)
        return vandermonde_prod


def _conj(x):
    return torch.cat([x, x.conj()], dim=-1)

if tuple(map(int, torch.__version__.split(".")[:2])) >= (1, 10):

    def _resolve_conj(x):
        return x.conj().resolve_conj()

else:

    def _resolve_conj(x):
        return x.conj()