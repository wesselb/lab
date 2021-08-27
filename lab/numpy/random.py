import warnings

import numpy as np

from . import dispatch, B, Numeric
from ..shape import unwrap_dimension
from ..types import NPDType, NPRandomState, Int

__all__ = []


@dispatch
def create_random_state(_: NPDType, seed: Int = 0):
    return np.random.RandomState(seed=seed)


def _warn_dtype(dtype):
    if B.issubdtype(dtype, np.integer):
        warnings.warn("Casting random number of type float to type integer.")


@dispatch
def rand(state: NPRandomState, dtype: NPDType, *shape: Int):
    _warn_dtype(dtype)
    return state, B.cast(dtype, state.rand(*shape))


@dispatch
def rand(dtype: NPDType, *shape: Int):
    return rand(np.random.random.__self__, dtype, *shape)[1]


@dispatch
def randn(state: NPRandomState, dtype: NPDType, *shape: Int):
    _warn_dtype(dtype)
    return state, B.cast(dtype, state.randn(*shape))


@dispatch
def randn(dtype: NPDType, *shape: Int):
    return randn(np.random.random.__self__, dtype, *shape)[1]


@dispatch
def choice(state: NPRandomState, a: Numeric, n: Int):
    inds = state.choice(unwrap_dimension(B.shape(a)[0]), n, replace=True)
    choices = a[inds]
    return state, choices[0] if n == 1 else choices


@dispatch
def choice(a: Numeric, n: Int):
    return choice(np.random.random.__self__, a, n)[1]
