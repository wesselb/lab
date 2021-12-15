import warnings

import numpy as np

from . import dispatch, B, Numeric
from ..shape import unwrap_dimension
from ..types import NPDType, NPRandomState, Int

__all__ = []


@dispatch
def create_random_state(_: NPDType, seed: Int = 0):
    return np.random.RandomState(seed=seed)


@dispatch
def global_random_state(_: NPDType):
    return np.random.random.__self__


@dispatch
def set_global_random_state(state: NPRandomState):
    np.random.random.__self__.set_state(state.get_state())


def _warn_dtype(dtype):
    if B.issubdtype(dtype, np.integer):
        warnings.warn("Casting random number of type float to type integer.")


@dispatch
def rand(state: NPRandomState, dtype: NPDType, *shape: Int):
    _warn_dtype(dtype)
    return state, B.cast(dtype, state.rand(*shape))


@dispatch
def rand(dtype: NPDType, *shape: Int):
    return rand(global_random_state(dtype), dtype, *shape)[1]


@dispatch
def randn(state: NPRandomState, dtype: NPDType, *shape: Int):
    _warn_dtype(dtype)
    return state, B.cast(dtype, state.randn(*shape))


@dispatch
def randn(dtype: NPDType, *shape: Int):
    return randn(global_random_state(dtype), dtype, *shape)[1]


@dispatch
def choice(state: NPRandomState, a: Numeric, n: Int):
    inds = state.choice(unwrap_dimension(B.shape(a)[0]), n, replace=True)
    choices = a[inds]
    return state, choices[0] if n == 1 else choices


@dispatch
def choice(a: Numeric, n: Int):
    return choice(global_random_state(a), a, n)[1]


@dispatch
def randint(state: NPRandomState, dtype: NPDType, *shape: Int, lower: Int = 0, upper):
    dtype = B.dtype_int(dtype)
    return state, state.randint(lower, upper, shape, dtype=dtype)


@dispatch
def randint(dtype: NPDType, *shape: Int, lower: Int = 0, upper):
    state = global_random_state(dtype)
    return randint(state, dtype, *shape, lower=lower, upper=upper)[1]
