import autograd.numpy as anp

from . import dispatch, Numeric
from ..types import Int

__all__ = []


@dispatch(Numeric)
def length(a):
    return anp.size(a)


@dispatch(Numeric)
def expand_dims(a, axis=0):
    return anp.expand_dims(a, axis=axis)


@dispatch(Numeric)
def squeeze(a):
    return anp.squeeze(a)


@dispatch(Numeric)
def diag(a):
    return anp.diag(a)


@dispatch([Numeric])
def stack(*elements, **kw_args):
    return anp.stack(elements, axis=kw_args.get("axis", 0))


@dispatch(Numeric)
def unstack(a, axis=0):
    out = anp.split(a, anp.arange(1, a.shape[axis]), axis)
    return [x.squeeze(axis=axis) for x in out]


@dispatch(Numeric, [Int])
def reshape(a, *shape):
    return anp.reshape(a, shape)


@dispatch([Numeric])
def concat(*elements, **kw_args):
    return anp.concatenate(elements, axis=kw_args.get("axis", 0))


@dispatch(Numeric, [Int])
def tile(a, *repeats):
    return anp.tile(a, repeats)
