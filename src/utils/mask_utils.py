"""Batch-element-wise masking utilities."""

import torch
from torch import Tensor


def maskmax(t: Tensor, m: Tensor, dim: int, **kwargs):
    """Get the max element along dim, ignoring elements not in mask.

    Examples
    --------
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6]])
    >>> m = torch.tensor([[1, 1, 0], [1, 1, 0]]).bool()
    >>> maskmax(t, m, 1) # tensor([2., 5.])

    """
    return torch.amax(torch.where(m, t, -torch.inf), dim=dim, **kwargs)


def maskmin(t: Tensor, m: Tensor, dim: int, **kwargs):
    """Get the min element along dim, ignoring elements not in mask.

    Examples
    --------
    >>> t = torch.tensor([[3, 2, 1], [6, 5, 4]])
    >>> m = torch.tensor([[1, 1, 0], [1, 1, 0]]).bool()
    >>> maskmin(t, m, 1) # tensor([2., 5.])

    """
    return torch.amin(torch.where(m, t, +torch.inf), dim=dim, **kwargs)


def maskmean(t: Tensor, m: Tensor, dim: int, **kwargs):
    """Get the mean along dim, ignoring elements not in mask.

    Examples
    --------
    >>> t = torch.tensor([[1, 1, 9], [2, 2, 9]])
    >>> m = torch.tensor([[1, 1, 0], [1, 1, 0]]).bool()
    >>> maskmean(t, m, 1) # tensor([1., 2.])

    """
    return torch.nanmean(torch.where(m, t, torch.nan), dim=dim, **kwargs)


def masklast(t: Tensor, m: Tensor, dim: int, *, keepdim: bool = False):
    """Get the last element along dim, ignoring elements not in mask.

    Examples
    --------
    >>> t = torch.tensor([[2, 1, 3], [4, 1, 6]])
    >>> m = torch.tensor([[1, 1, 1], [1, 1, 0]]).bool()
    >>> masklast(t, m, 1) # tensor([3., 1.])

    """
    indices = (torch.sum(m, dim) - 1).int()
    idx = [torch.arange(0, size) for size in indices.shape]
    idx.insert(dim, indices)
    t = t[idx]
    if keepdim:
        t = t.unsqueeze(dim)

    return t


def maskffill(t: Tensor, m: Tensor, dim: int):
    """Forward-fill values not in mask."""
    indices = torch.cummax(m, dim).indices
    return t.gather(dim, indices)


def maskbfill(t: Tensor, m: Tensor, dim: int):
    """Backward-fill values not in mask."""
    indices = torch.cummax(m.flip(dim), dim).indices
    return t.flip(dim).gather(dim, indices).flip(dim)


def maskroll(m: Tensor, *tensors: Tensor):
    """Roll elements specified by `mask` to beginning over last dimension of mask, for all tensors.

    Returns
    -------
        tuple of new mask and new tensors.

    """
    nkeep = m.sum(-1, keepdim=True)
    arange = torch.arange(m.size(-1), device=m.device).expand_as(m)
    m_ = nkeep > arange
    tensors_: list[Tensor] = []
    for t in tensors:
        t_ = torch.zeros_like(t)
        t_[m_] = t[m]
        tensors_.append(t_)

    return m_, *tensors_
