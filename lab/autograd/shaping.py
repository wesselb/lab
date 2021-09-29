import autograd.numpy as anp
from plum import Union

from . import dispatch, Numeric
from ..types import Int

__all__ = []


@dispatch
def length(a: Numeric):
    return anp.size(a)


@dispatch
def _expand_dims(a: Numeric, axis: Int = 0):
    return anp.expand_dims(a, axis=axis)


@dispatch
def squeeze(a: Numeric, axis: Union[Int, None] = None):
    return anp.squeeze(a, axis=axis)


@dispatch
def broadcast_to(a: Numeric, *shape: Int):
    return anp.broadcast_to(a, shape)


@dispatch
def diag(a: Numeric):
    return anp.diag(a)


@dispatch
def diag_extract(a: Numeric):
    return anp.diagonal(a, axis1=-2, axis2=-1)


@dispatch
def stack(*elements: Numeric, axis: Int = 0):
    return anp.stack(elements, axis=axis)


@dispatch
def _unstack(a: Numeric, axis: Int = 0):
    out = anp.split(a, anp.arange(1, a.shape[axis]), axis)
    return [x.squeeze(axis=axis) for x in out]


@dispatch
def reshape(a: Numeric, *shape: Int):
    return anp.reshape(a, shape)


@dispatch
def concat(*elements: Numeric, axis: Int = 0):
    return anp.concatenate(elements, axis=axis)


@dispatch
def tile(a: Numeric, *repeats: Int):
    return anp.tile(a, repeats)
