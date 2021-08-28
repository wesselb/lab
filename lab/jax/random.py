import jax
from plum import Dispatcher

from . import B, dispatch
from ..types import Int, JAXDType, JAXNumeric, JAXRandomState

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
def choice(state: JAXRandomState, a: JAXNumeric, n: Int):
    state, key = jax.random.split(state)
    inds = jax.random.choice(key, a.shape[0], (n,), replace=True)
    choices = a[inds]
    return state, choices[0] if n == 1 else choices


@dispatch
def choice(a: JAXNumeric, n: Int):
    state, res = choice(global_random_state(a), a, n)
    B.jax_global_random_state = state
    return res
