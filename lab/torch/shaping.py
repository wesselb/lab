# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from . import dispatch, Torch

__all__ = []


@dispatch(Torch)
def shape(a):
    s = a.shape
    return tuple(s[i] for i in range(rank(a)))


@dispatch(Torch)
def shape_int(a):
    return shape(a)


@dispatch(Torch)
def rank(a):
    return len(a.shape)


@dispatch(Torch)
def length(a):
    return a.numel()


@dispatch(Torch)
def expand_dims(a, axis=0):
    return torch.unsqueeze(a, dim=axis)


@dispatch(Torch)
def diag(a):
    return torch.diag(a)


@dispatch(Torch)
def vec_to_tril(a):
    if rank(a) != 1:
        raise ValueError('Input must be rank 1.')

    # Figure out shape of output.
    n = shape_int(a)[0]
    m = int(((1 + 8 * n) ** .5 - 1) / 2)

    # Construct output and return.
    out = torch.zeros(m, m, dtype=a.dtype)
    out[np.tril_indices(m)] = a
    return out


@dispatch(Torch)
def tril_to_vec(a):
    if rank(a) != 2:
        raise ValueError('Input must be rank 2.')
    n, m = shape_int(a)
    if n != m:
        raise ValueError('Input must be square.')
    return a[np.tril_indices(n)]


# ----

@dispatch(Torch)
def reshape(a, shape=(-1,)):
    return torch.reshape(a, shape=shape)
