import warnings

import numpy as np

from . import dispatch, B, Numeric
from ..shape import unwrap_dimension
from ..types import NPDType, Int

__all__ = []


def _warn_dtype(dtype):
    if B.issubdtype(dtype, np.integer):
        warnings.warn("Casting random number of type float to type integer.")


@dispatch
def rand(dtype: NPDType, *shape: Int):
    _warn_dtype(dtype)
    return B.cast(dtype, np.random.rand(*shape))


@dispatch
def randn(dtype: NPDType, *shape: Int):
    _warn_dtype(dtype)
    return B.cast(dtype, np.random.randn(*shape))


@dispatch
def choice(a: Numeric, n: Int):
    inds = np.random.choice(unwrap_dimension(B.shape(a)[0]), n, replace=True)
    choices = a[inds]
    return choices[0] if n == 1 else choices
