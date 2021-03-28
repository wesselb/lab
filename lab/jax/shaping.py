import jax.numpy as jnp

from . import dispatch, Numeric
from ..types import Int

__all__ = []


@dispatch(Numeric)
def length(a):
    return jnp.size(a)


@dispatch(Numeric)
def expand_dims(a, axis=0):
    return jnp.expand_dims(a, axis=axis)


@dispatch(Numeric)
def squeeze(a):
    return jnp.squeeze(a)


@dispatch(Numeric)
def diag(a):
    return jnp.diag(a)


@dispatch(Numeric)
def diag_extract(a):
    return jnp.diagonal(a, axis1=-2, axis2=-1)


@dispatch([Numeric])
def stack(*elements, axis=0):
    return jnp.stack(elements, axis=axis)


@dispatch(Numeric)
def unstack(a, axis=0):
    out = jnp.split(a, jnp.arange(1, a.shape[axis]), axis)
    return [x.squeeze(axis=axis) for x in out]


@dispatch(Numeric, [Int])
def reshape(a, *shape):
    return jnp.reshape(a, shape)


@dispatch([Numeric])
def concat(*elements, axis=0):
    return jnp.concatenate(elements, axis=axis)


@dispatch(Numeric, [Int])
def tile(a, *repeats):
    return jnp.tile(a, repeats)
