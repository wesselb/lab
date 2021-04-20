import numpy as np

from . import dispatch, Numeric
from ..types import Int

__all__ = []


@dispatch
def length(a: Numeric):
    return np.size(a)


@dispatch
def expand_dims(a: Numeric, axis=0):
    return np.expand_dims(a, axis=axis)


@dispatch
def squeeze(a: Numeric):
    return np.squeeze(a)


@dispatch
def diag(a: Numeric):
    return np.diag(a)


@dispatch
def diag_extract(a: Numeric):
    return np.diagonal(a, axis1=-2, axis2=-1)


@dispatch
def stack(*elements: Numeric, axis=0):
    return np.stack(elements, axis=axis)


@dispatch
def unstack(a: Numeric, axis=0):
    out = np.split(a, np.arange(1, a.shape[axis]), axis)
    return [x.squeeze(axis=axis) for x in out]


@dispatch
def reshape(a: Numeric, *shape: Int):
    return np.reshape(a, shape)


@dispatch
def concat(*elements: Numeric, axis=0):
    return np.concatenate(elements, axis=axis)


@dispatch
def tile(a: Numeric, *repeats: Int):
    return np.tile(a, repeats)
