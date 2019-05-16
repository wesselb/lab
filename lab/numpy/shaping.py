# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import autograd.numpy as np

from . import dispatch
from ..shaping import _vec_to_tril_shape
from ..types import NPNumeric, NPDimension

__all__ = []


@dispatch(NPNumeric)
def shape(a):
    return np.shape(a)


@dispatch(NPNumeric)
def shape_int(a):
    return np.shape(a)


@dispatch(NPNumeric)
def rank(a):
    if hasattr(a, 'ndim'):
        return a.ndim
    else:
        return len(shape(a))


@dispatch(NPNumeric)
def length(a):
    return np.size(a)


@dispatch(NPNumeric)
def expand_dims(a, axis=0):
    return np.expand_dims(a, axis=axis)


@dispatch(NPNumeric)
def squeeze(a):
    return np.squeeze(a)


@dispatch(NPNumeric)
def diag(a):
    return np.diag(a)


@dispatch(NPNumeric)
def vec_to_tril(a):
    if rank(a) != 1:
        raise ValueError('Input must be rank 1.')
    m = _vec_to_tril_shape(a)
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


@dispatch([NPNumeric])
def stack(*elements, **kw_args):
    return np.stack(elements, axis=kw_args.get('axis', 0))


@dispatch(NPNumeric)
def unstack(a, axis=0):
    out = np.split(a, np.arange(1, a.shape[axis]), axis)
    return [x.squeeze(axis=axis) for x in out]


@dispatch(NPNumeric, [NPDimension])
def reshape(a, *shape):
    return np.reshape(a, shape)


@dispatch([NPNumeric])
def concat(*elements, **kw_args):
    return np.concatenate(elements, axis=kw_args.get('axis', 0))


@dispatch(NPNumeric, object)
def take(a, indices, axis=0):
    return np.take(a, indices, axis=axis)