import numpy as np

from . import dispatch, Numeric
from ..types import Int

__all__ = []


@dispatch(Numeric)
def length(a):
    return np.size(a)


@dispatch(Numeric)
def expand_dims(a, axis=0):
    return np.expand_dims(a, axis=axis)


@dispatch(Numeric)
def squeeze(a):
    return np.squeeze(a)


@dispatch(Numeric)
def diag(a):
    return np.diag(a)


@dispatch(Numeric)
def diag_extract(a):
    return np.diagonal(a, axis1=-2, axis2=-1)


@dispatch([Numeric])
def stack(*elements, axis=0):
    return np.stack(elements, axis=axis)


@dispatch(Numeric)
def unstack(a, axis=0):
    out = np.split(a, np.arange(1, a.shape[axis]), axis)
    return [x.squeeze(axis=axis) for x in out]


@dispatch(Numeric, [Int])
def reshape(a, *shape):
    return np.reshape(a, shape)


@dispatch([Numeric])
def concat(*elements, axis=0):
    return np.concatenate(elements, axis=axis)


@dispatch(Numeric, [Int])
def tile(a, *repeats):
    return np.tile(a, repeats)
