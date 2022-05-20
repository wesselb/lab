import warnings

import numpy as np
from plum import Union

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
def choice(state: NPRandomState, a: Numeric, n: Int, *, p: Union[Numeric, None] = None):
    # Probabilities must sum to one.
    if p is not None:
        p = p / np.sum(p, axis=0, keepdims=True)
    # Feeding `a` to `choice` will not work if `a` is higher-dimensional.
    inds = state.choice(unwrap_dimension(B.shape(a)[0]), n, replace=True, p=p)
    choices = a[inds]
    return state, choices


@dispatch
def choice(a: Numeric, *shape: Int, p: Union[Numeric, None] = None):
    return choice(global_random_state(a), a, *shape, p=p)[1]


@dispatch
def randint(state: NPRandomState, dtype: NPDType, *shape: Int, lower: Int = 0, upper):
    dtype = B.dtype_int(dtype)
    return state, state.randint(lower, upper, shape, dtype=dtype)


@dispatch
def randint(dtype: NPDType, *shape: Int, lower: Int = 0, upper):
    state = global_random_state(dtype)
    return randint(state, dtype, *shape, lower=lower, upper=upper)[1]


@dispatch
def randperm(state: NPRandomState, dtype: NPDType, n: Int):
    dtype = B.dtype_int(dtype)
    return state, B.cast(dtype, state.permutation(n))


@dispatch
def randperm(dtype: NPDType, n: Int):
    return randperm(global_random_state(dtype), dtype, n)[1]
