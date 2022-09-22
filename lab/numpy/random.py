import warnings

import numpy as np
from plum import Union

from . import dispatch, B, Numeric
from ..types import NPDType, NPRandomState, Int
from ..util import broadcast_shapes

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
def randcat(state: NPRandomState, p: Numeric, n: Int):
    # Probabilities must sum to one.
    p = p / np.sum(p, axis=-1, keepdims=True)
    # Perform sampling routine.
    cdf = np.cumsum(p, axis=-1)
    u = state.rand(n, *p.shape[:-1])
    inds = np.sum(u[..., None] < cdf[None], axis=-1) - 1
    # Be sure to return the right data type.
    return state, B.cast(B.dtype_int(p), inds)


@dispatch
def randcat(p: Numeric, *shape: Int):
    return randcat(global_random_state(p), p, *shape)[1]


@dispatch
def choice(a: Numeric, *shape: Int, p: Union[Numeric, None] = None):
    # This method is necessary to break ambiguity.
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


@dispatch
def randgamma(
    state: NPRandomState,
    dtype: NPDType,
    *shape: Int,
    alpha: Numeric,
    scale: Numeric,
):
    _warn_dtype(dtype)
    shape = shape + broadcast_shapes(B.shape(alpha), B.shape(scale))
    return state, B.cast(dtype, state.gamma(alpha, size=shape) * scale)


@dispatch
def randgamma(dtype: NPDType, *shape: Int, alpha: Numeric, scale: Numeric):
    state = global_random_state(dtype)
    return randgamma(state, dtype, *shape, alpha=alpha, scale=scale)[1]
