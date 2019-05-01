# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import warnings
import numpy as np_
import autograd.numpy as np

from . import dispatch, B
from ..types import NPShape, NPDType

__all__ = []


def _warn_dtype(dtype):
    if B.issubdtype(dtype, np_.integer):
        warnings.warn('Casting random number of type float to type integer.')


@dispatch(NPShape, NPDType)
def rand(shape, dtype):
    _warn_dtype(dtype)
    return B.cast(np.random.rand(*shape), dtype)


@dispatch(NPShape, NPDType)
def randn(shape, dtype):
    _warn_dtype(dtype)
    return B.cast(np.random.randn(*shape), dtype)
