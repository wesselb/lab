import jax
import jax.numpy as jnp
from plum import Dispatcher, Union

from . import dispatch, B, Numeric
from ..types import Int, JAXDType, JAXNumeric, JAXRandomState
from ..util import broadcast_shapes

__all__ = []

_dispatch = Dispatcher()


@dispatch
def create_random_state(_: JAXDType, seed: Int = 0):
    return jax.random.PRNGKey(seed=seed)


B.jax_global_random_state = jax.random.PRNGKey(seed=0)


@dispatch
def global_random_state(_: JAXDType):
    return B.jax_global_random_state


@dispatch
def set_global_random_state(state: JAXRandomState):
    B.jax_global_random_state = state


@dispatch
def rand(state: JAXRandomState, dtype: JAXDType, *shape: Int):
    state, key = jax.random.split(state)
    return state, B.to_active_device(jax.random.uniform(key, shape, dtype=dtype))


@dispatch
def rand(dtype: JAXDType, *shape: Int):
    state, res = rand(global_random_state(dtype), dtype, *shape)
    B.jax_global_random_state = state
    return res


@dispatch
def randn(state: JAXRandomState, dtype: JAXDType, *shape: Int):
    state, key = jax.random.split(state)
    return state, B.to_active_device(jax.random.normal(key, shape, dtype=dtype))


@dispatch
def randn(dtype: JAXDType, *shape: Int):
    state, res = randn(global_random_state(dtype), dtype, *shape)
    B.jax_global_random_state = state
    return res


@dispatch
def randcat(state: JAXRandomState, p: JAXNumeric, *shape: Int):
    state, key = jax.random.split(state)
    # We need to tile to make the batching work.
    p = B.tile(
        B.expand_dims(p, axis=0, times=len(shape)),
        *shape,
        *((1,) * len(p.shape)),
    )
    inds = jax.random.categorical(key, jnp.log(p))
    return state, inds


@dispatch
def randcat(p: JAXNumeric, *shape: Int):
    state, res = randcat(global_random_state(p), p, *shape)
    B.jax_global_random_state = state
    return res


@dispatch
def choice(a: JAXNumeric, *shape: Int, p: Union[Numeric, None] = None):
    # This method is necessary to break ambiguity.
    state, res = choice(global_random_state(a), a, *shape, p=p)
    B.jax_global_random_state = state
    return res


@dispatch
def randint(
    state: JAXRandomState,
    dtype: JAXDType,
    *shape: Int,
    lower: Int = 0,
    upper: Int,
):
    dtype = B.dtype_int(dtype)
    state, key = jax.random.split(state)
    return state, B.to_active_device(
        jax.random.randint(key, shape, lower, upper, dtype=dtype)
    )


@dispatch
def randint(
    dtype: JAXDType,
    *shape: Int,
    lower: Int = 0,
    upper: Int,
):
    state, res = randint(
        global_random_state(dtype),
        dtype,
        *shape,
        lower=lower,
        upper=upper,
    )
    B.jax_global_random_state = state
    return res


@dispatch
def randperm(state: JAXRandomState, dtype: JAXDType, n: Int):
    dtype = B.dtype_int(dtype)
    state, key = jax.random.split(state)
    return state, B.to_active_device(B.cast(dtype, jax.random.permutation(key, n)))


@dispatch
def randperm(dtype: JAXDType, n: Int):
    state, res = randperm(global_random_state(dtype), dtype, n)
    B.jax_global_random_state = state
    return res


@dispatch
def randgamma(
    state: JAXRandomState,
    dtype: JAXDType,
    *shape: Int,
    alpha: Numeric,
    scale: Numeric,
):
    state, key = jax.random.split(state)
    shape = shape + broadcast_shapes(B.shape(alpha), B.shape(scale))
    sample = B.to_active_device(jax.random.gamma(key, alpha, shape, dtype=dtype))
    sample = sample * B.to_active_device(B.cast(dtype, scale))
    return state, sample


@dispatch
def randgamma(dtype: JAXDType, *shape: Int, alpha: Numeric, scale: Numeric):
    state = global_random_state(dtype)
    state, res = randgamma(state, dtype, *shape, alpha=alpha, scale=scale)
    B.jax_global_random_state = state
    return res
