import warnings

import numpy as np

from . import dispatch, B, Numeric
from ..types import NPDType, Int

__all__ = []


def _warn_dtype(dtype):
    if B.issubdtype(dtype, np.integer):
        warnings.warn('Casting random number of type float to type integer.')


@dispatch(NPDType, [Int])
def rand(dtype, *shape):
    _warn_dtype(dtype)
    return B.cast(dtype, np.random.rand(*shape))


@dispatch(NPDType, [Int])
def randn(dtype, *shape):
    _warn_dtype(dtype)
    return B.cast(dtype, np.random.randn(*shape))


@dispatch(Numeric, Int)
def choice(a, n):
    inds = np.random.choice(a.shape[0], n, replace=True)
    choices = a[inds]
    return choices[0] if n == 1 else choices
