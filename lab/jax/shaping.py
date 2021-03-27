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


@dispatch([Numeric])
def stack(*elements, **kw_args):
    return jnp.stack(elements, axis=kw_args.get("axis", 0))


@dispatch(Numeric)
def unstack(a, axis=0):
    out = jnp.split(a, jnp.arange(1, a.shape[axis]), axis)
    return [x.squeeze(axis=axis) for x in out]


@dispatch(Numeric, [Int])
def reshape(a, *shape):
    return jnp.reshape(a, shape)


@dispatch([Numeric])
def concat(*elements, **kw_args):
    return jnp.concatenate(elements, axis=kw_args.get("axis", 0))


@dispatch(Numeric, [Int])
def tile(a, *repeats):
    return jnp.tile(a, repeats)
