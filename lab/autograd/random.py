# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import warnings

import autograd.numpy as anp
import numpy as np

from . import dispatch, B
from ..types import NPDimension, NPDType

__all__ = []


def _warn_dtype(dtype):
    if B.issubdtype(dtype, np.integer):
        warnings.warn('Casting random number of type float to type integer.')


@dispatch(NPDType, [NPDimension])
def rand(dtype, *shape):
    _warn_dtype(dtype)
    return B.cast(dtype, anp.random.rand(*shape))


@dispatch(NPDType, [NPDimension])
def randn(dtype, *shape):
    _warn_dtype(dtype)
    return B.cast(dtype, anp.random.randn(*shape))
