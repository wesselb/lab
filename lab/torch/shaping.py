# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import torch

from . import dispatch
from ..types import TorchNumeric, TorchListOrTuple, ListOrTuple

__all__ = []


@dispatch(TorchNumeric)
def shape(a):
    s = a.shape
    return tuple(s[i] for i in range(rank(a)))


@dispatch(TorchNumeric)
def shape_int(a):
    return shape(a)


@dispatch(TorchNumeric)
def rank(a):
    return len(a.shape)


@dispatch(TorchNumeric)
def length(a):
    return a.numel()


@dispatch(TorchNumeric)
def expand_dims(a, axis=0):
    return torch.unsqueeze(a, dim=axis)


@dispatch(TorchNumeric)
def diag(a):
    return torch.diag(a)


@dispatch(TorchNumeric)
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


@dispatch(TorchNumeric)
def tril_to_vec(a):
    if rank(a) != 2:
        raise ValueError('Input must be rank 2.')
    n, m = shape_int(a)
    if n != m:
        raise ValueError('Input must be square.')
    return a[np.tril_indices(n)]


@dispatch(TorchListOrTuple)
def stack(a, axis=0):
    return torch.stack(a, dim=axis)


@dispatch(TorchNumeric)
def unstack(a, axis=0):
    return torch.unbind(a, dim=axis)


@dispatch(TorchNumeric)
def reshape(a, shape=(-1,)):
    return torch.reshape(a, shape=shape)


@dispatch(TorchListOrTuple)
def concat(a, axis=0):
    return torch.cat(a, dim=axis)


@dispatch(TorchNumeric, ListOrTuple)
def take(a, indices, axis=0):
    if axis > 0:
        a = torch.transpose(a, 0, axis)
    a = a[(indices,) + (slice(None),) * (rank(a) - 1)]
    if axis > 0:
        a = torch.transpose(a, 0, axis)
    return a
