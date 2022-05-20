from typing import Optional, Union

import opt_einsum as oe
import torch

from . import dispatch, B, Numeric
from .custom import torch_register
from ..custom import toeplitz_solve, s_toeplitz_solve, expm, s_expm, logm, s_logm
from ..linear_algebra import _default_perm
from ..types import Int

__all__ = []


@dispatch
def matmul(a: Numeric, b: Numeric, tr_a: bool = False, tr_b: bool = False):
    a = transpose(a) if tr_a else a
    b = transpose(b) if tr_b else b
    return torch.matmul(a, b)


@dispatch
def einsum(equation: str, *elements: Numeric):
    return oe.contract(equation, *elements, backend="torch")


@dispatch
def transpose(a: Numeric, perm: Optional[Union[tuple, list]] = None):
    # Correctly handle special cases.
    rank_a = B.rank(a)
    if rank_a == 0:
        return a
    elif rank_a == 1 and perm is None:
        return a[None, :]

    if perm is None:
        perm = _default_perm(a)
    return a.permute(*perm)


@dispatch
def trace(a: Numeric, axis1: Int = -2, axis2: Int = -1):
    return torch.sum(torch.diagonal(a, dim1=axis1, dim2=axis2), dim=-1)


@dispatch
def svd(a: Numeric, compute_uv: bool = True):
    u, s, v = torch.linalg.svd(a, full_matrices=False)
    return (u, s, v) if compute_uv else s


@dispatch
def eig(a: Numeric, compute_eigvecs: bool = True):
    vals, vecs = torch.linalg.eig(a)
    return (vals, vecs) if compute_eigvecs else vals


@dispatch
def solve(a: Numeric, b: Numeric):
    return torch.linalg.solve(a, b)


@dispatch
def inv(a: Numeric):
    return torch.inverse(a)


@dispatch
def det(a: Numeric):
    return torch.linalg.det(a)


@dispatch
def logdet(a: Numeric):
    return torch.logdet(a)


_expm = torch_register(expm, s_expm)


@dispatch
def expm(a: Numeric):
    return _expm(a)


_logm = torch_register(logm, s_logm)


@dispatch
def logm(a: Numeric):
    return _logm(a)


@dispatch
def _cholesky(a: Numeric):
    return torch.linalg.cholesky(a)


@dispatch
def cholesky_solve(a: Numeric, b: Numeric):
    return torch.cholesky_solve(b, a, upper=False)


@dispatch
def triangular_solve(a: Numeric, b: Numeric, lower_a: bool = True):
    return torch.linalg.solve_triangular(a, b, upper=not lower_a)


_toeplitz_solve = torch_register(toeplitz_solve, s_toeplitz_solve)


@dispatch
def toeplitz_solve(a: Numeric, b: Numeric, c: Numeric):
    return _toeplitz_solve(a, b, c)
