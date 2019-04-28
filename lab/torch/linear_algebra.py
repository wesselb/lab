# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import torch

from . import dispatch, Torch, B

__all__ = []


@dispatch(Torch, Torch)
def matmul(a, b, tr_a=False, tr_b=False):
    a = a.t() if tr_a else a
    b = b.t() if tr_b else b
    return torch.matmul(a, b)


@dispatch(Torch)
def transpose(a):
    return a.t()


@dispatch(Torch)
def trace(a, axis1=0, axis2=1):
    return torch.sum(torch.diagonal(a, dim1=axis1, dim2=axis2), dim=-1)


@dispatch(Torch, Torch)
def kron(a, b):
    shape_a = B.shape_int(a)
    shape_b = B.shape_int(b)

    # Check that ranks are equal.
    if len(shape_a) != len(shape_b):
        raise ValueError('Inputs must have equal rank.')

    a = a.view(*sum([[i, 1] for i in shape_a], []))
    b = b.view(*sum([[1, i] for i in shape_b], []))
    return torch.reshape(a * b, tuple(x * y for x, y in zip(shape_a, shape_b)))


@dispatch(Torch)
def svd(a, compute_uv=True):
    u, s, v = torch.svd(a, compute_uv=compute_uv)
    return (u, s, v) if compute_uv else s


@dispatch(Torch)
def cholesky(a):
    return torch.cholesky(a, upper=False)
