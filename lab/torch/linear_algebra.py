import torch

from . import dispatch, B, Numeric
from .custom import torch_register
from ..custom import toeplitz_solve, s_toeplitz_solve, expm, s_expm, logm, s_logm
from ..linear_algebra import _default_perm
from ..shape import unwrap_dimension

__all__ = []


@dispatch
def matmul(a: Numeric, b: Numeric, tr_a=False, tr_b=False):
    a = transpose(a) if tr_a else a
    b = transpose(b) if tr_b else b
    return torch.matmul(a, b)


@dispatch
def transpose(a: Numeric, perm=None):
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
def trace(a: Numeric, axis1=-2, axis2=-1):
    return torch.sum(torch.diagonal(a, dim1=axis1, dim2=axis2), dim=-1)


@dispatch
def kron(a: Numeric, b: Numeric):
    shape_a = B.shape(a)
    shape_b = B.shape(b)

    # Check that ranks are equal.
    if len(shape_a) != len(shape_b):
        raise ValueError("Inputs must have equal rank.")

    a = a.view(*sum([[unwrap_dimension(i), 1] for i in shape_a], []))
    b = b.view(*sum([[1, unwrap_dimension(i)] for i in shape_b], []))
    return torch.reshape(a * b, tuple(x * y for x, y in zip(shape_a, shape_b)))


@dispatch
def svd(a: Numeric, compute_uv=True):
    u, s, v = torch.svd(a, compute_uv=compute_uv)
    return (u, s, v) if compute_uv else s


@dispatch
def solve(a: Numeric, b: Numeric):
    return torch.solve(b, a)[0]


@dispatch
def inv(a: Numeric):
    return torch.inverse(a)


@dispatch
def det(a: Numeric):
    return torch.det(a)


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
def cholesky(a: Numeric):
    return torch.cholesky(a, upper=False)


@dispatch
def cholesky_solve(a: Numeric, b: Numeric):
    return torch.cholesky_solve(b, a, upper=False)


@dispatch
def triangular_solve(a: Numeric, b: Numeric, lower_a=True):
    return torch.triangular_solve(b, a, upper=not lower_a)[0]


_toeplitz_solve = torch_register(toeplitz_solve, s_toeplitz_solve)


@dispatch
def toeplitz_solve(a: Numeric, b: Numeric, c: Numeric):
    return _toeplitz_solve(a, b, c)
