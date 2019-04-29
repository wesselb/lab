# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from . import dispatch
from ..types import NPNumeric, NPListOrTuple

__all__ = []


@dispatch(NPNumeric)
def shape(a):
    return np.shape(a)


@dispatch(NPNumeric)
def shape_int(a):
    return np.shape(a)


@dispatch(NPNumeric)
def rank(a):
    return a.ndim


@dispatch(NPNumeric)
def length(a):
    return np.size(a)


@dispatch(NPNumeric)
def expand_dims(a, axis=0):
    return np.expand_dims(a, axis=axis)


@dispatch(NPNumeric)
def diag(a):
    return np.diag(a)


@dispatch(NPNumeric)
def vec_to_tril(a):
    if rank(a) != 1:
        raise ValueError('Input must be rank 1.')

    # Figure out shape of output.
    n = shape_int(a)[0]
    m = int(((1 + 8 * n) ** .5 - 1) / 2)

    # Construct output and return.
    out = np.zeros((m, m), dtype=a.dtype)
    out[np.tril_indices(m)] = a
    return out


@dispatch(NPNumeric)
def tril_to_vec(a):
    if rank(a) != 2:
        raise ValueError('Input must be rank 2.')
    n, m = shape_int(a)
    if n != m:
        raise ValueError('Input must be square.')
    return a[np.tril_indices(n)]


@dispatch(NPListOrTuple)
def stack(a, axis=0):
    return np.stack(a, axis=axis)


# ----

@dispatch(NPNumeric)
def reshape(a, shape=(-1,)):
    return np.reshape(a, newshape=shape)
