# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import torch

from . import dispatch, B
from .custom import torch_register
from ..custom import toeplitz_solve, s_toeplitz_solve
from ..linear_algebra import _default_perm
from ..types import TorchNumeric

__all__ = []


@dispatch(TorchNumeric, TorchNumeric)
def matmul(a, b, tr_a=False, tr_b=False):
    a = a.t() if tr_a else a
    b = b.t() if tr_b else b
    return torch.matmul(a, b)


@dispatch(TorchNumeric)
def transpose(a, perm=None):
    if perm is None:
        perm = _default_perm(a)
    return a.permute(*perm)


@dispatch(TorchNumeric)
def trace(a, axis1=0, axis2=1):
    return torch.sum(torch.diagonal(a, dim1=axis1, dim2=axis2), dim=-1)


@dispatch(TorchNumeric, TorchNumeric)
def kron(a, b):
    shape_a = B.shape_int(a)
    shape_b = B.shape_int(b)

    # Check that ranks are equal.
    if len(shape_a) != len(shape_b):
        raise ValueError('Inputs must have equal rank.')

    a = a.view(*sum([[i, 1] for i in shape_a], []))
    b = b.view(*sum([[1, i] for i in shape_b], []))
    return torch.reshape(a * b, tuple(x * y for x, y in zip(shape_a, shape_b)))


@dispatch(TorchNumeric)
def svd(a, compute_uv=True):
    u, s, v = torch.svd(a, compute_uv=compute_uv)
    return (u, s, v) if compute_uv else s


@dispatch(TorchNumeric, TorchNumeric)
def solve(a, b):
    return torch.solve(b, a)[0]


@dispatch(TorchNumeric)
def inv(a):
    return torch.inverse(a)


@dispatch(TorchNumeric)
def det(a):
    return torch.det(a)


@dispatch(TorchNumeric)
def logdet(a):
    return torch.logdet(a)


@dispatch(TorchNumeric)
def cholesky(a):
    return torch.cholesky(a, upper=False)


@dispatch(TorchNumeric, TorchNumeric)
def cholesky_solve(a, b):
    return torch.cholesky_solve(b, a, upper=False)


@dispatch(TorchNumeric, TorchNumeric)
def triangular_solve(a, b, lower_a=True):
    return torch.triangular_solve(b, a, upper=not lower_a)[0]


f = torch_register(toeplitz_solve, s_toeplitz_solve)
dispatch(TorchNumeric, TorchNumeric, TorchNumeric)(f)
