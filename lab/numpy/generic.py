# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import autograd.numpy as np

from . import dispatch, B
from ..types import NPNumeric, NPDType, NPDimension

__all__ = []


@dispatch(NPNumeric)
def isnan(a):
    return np.isnan(a)


@dispatch(NPDType, [NPDimension])
def zeros(dtype, *shape):
    return np.zeros(shape, dtype=dtype)


@dispatch(NPDType, [NPDimension])
def ones(dtype, *shape):
    return np.ones(shape, dtype=dtype)


@dispatch(NPDType, NPDimension, NPDimension)
def eye(dtype, *shape):
    return np.eye(shape[0], shape[1], dtype=dtype)


@dispatch(NPNumeric, NPNumeric, int)
def linspace(a, b, num):
    return np.linspace(a, b, num, dtype=B.dtype(a))


@dispatch(NPNumeric, NPDType)
def cast(a, dtype):
    if hasattr(a, 'astype'):
        return a.astype(dtype)
    else:
        return np.array(a, dtype=dtype)


@dispatch(NPNumeric)
def identity(a):
    return np.array(a)


@dispatch(NPNumeric)
def abs(a):
    return np.abs(a)


@dispatch(NPNumeric)
def sign(a):
    return np.sign(a)


@dispatch(NPNumeric)
def sqrt(a):
    return np.sqrt(a)


@dispatch(NPNumeric)
def exp(a):
    return np.exp(a)


@dispatch(NPNumeric)
def log(a):
    return np.log(a)


@dispatch(NPNumeric)
def sin(a):
    return np.sin(a)


@dispatch(NPNumeric)
def cos(a):
    return np.cos(a)


@dispatch(NPNumeric)
def tan(a):
    return np.tan(a)


@dispatch(NPNumeric)
def tanh(a):
    return np.tanh(a)


@dispatch(NPNumeric, NPNumeric)
def add(a, b):
    return np.add(a, b)


@dispatch(NPNumeric, NPNumeric)
def subtract(a, b):
    return np.subtract(a, b)


@dispatch(NPNumeric, NPNumeric)
def multiply(a, b):
    return np.multiply(a, b)


@dispatch(NPNumeric, NPNumeric)
def divide(a, b):
    return np.divide(a, b)


@dispatch(NPNumeric, NPNumeric)
def power(a, b):
    return np.power(a, b)


@dispatch(NPNumeric, NPNumeric)
def minimum(a, b):
    return np.minimum(a, b)


@dispatch(NPNumeric, NPNumeric)
def maximum(a, b):
    return np.maximum(a, b)


@dispatch(NPNumeric)
def min(a, axis=None):
    return np.min(a, axis=axis)


@dispatch(NPNumeric)
def max(a, axis=None):
    return np.max(a, axis=axis)


@dispatch(NPNumeric)
def sum(a, axis=None):
    return np.sum(a, axis=axis)


@dispatch(NPNumeric)
def mean(a, axis=None):
    return np.mean(a, axis=axis)


@dispatch(NPNumeric)
def std(a, axis=None):
    return np.std(a, axis=axis, ddof=0)


@dispatch(NPNumeric)
def all(a, axis=None):
    return np.all(a, axis=axis)


@dispatch(NPNumeric)
def any(a, axis=None):
    return np.any(a, axis=axis)


@dispatch(NPNumeric, NPNumeric)
def lt(a, b):
    return np.less(a, b)


@dispatch(NPNumeric, NPNumeric)
def le(a, b):
    return np.less_equal(a, b)


@dispatch(NPNumeric, NPNumeric)
def gt(a, b):
    return np.greater(a, b)


@dispatch(NPNumeric, NPNumeric)
def ge(a, b):
    return np.greater_equal(a, b)
