import jax.numpy as jnp
from plum import Union

from . import dispatch, Numeric
from ..types import Int

__all__ = []


@dispatch
def length(a: Numeric):
    return jnp.size(a)


@dispatch
def _expand_dims(a: Numeric, axis: Int = 0):
    return jnp.expand_dims(a, axis=axis)


@dispatch
def squeeze(a: Numeric, axis: Union[Int, None] = None):
    return jnp.squeeze(a, axis=axis)


@dispatch
def broadcast_to(a: Numeric, *shape: Int):
    return jnp.broadcast_to(a, shape)


@dispatch
def diag(a: Numeric):
    return jnp.diag(a)


@dispatch
def diag_extract(a: Numeric):
    return jnp.diagonal(a, axis1=-2, axis2=-1)


@dispatch
def stack(*elements: Numeric, axis: Int = 0):
    return jnp.stack(elements, axis=axis)


@dispatch
def _unstack(a: Numeric, axis: Int = 0):
    out = jnp.split(a, jnp.arange(1, a.shape[axis]), axis)
    return [x.squeeze(axis=axis) for x in out]


@dispatch
def reshape(a: Numeric, *shape: Int):
    return jnp.reshape(a, shape)


@dispatch
def concat(*elements: Numeric, axis: Int = 0):
    return jnp.concatenate(elements, axis=axis)


@dispatch
def tile(a: Numeric, *repeats: Int):
    return jnp.tile(a, repeats)
