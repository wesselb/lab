# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from . import B, dispatch
from ..custom import bvn_cdf
from ..types import NPNumeric, NPDType, Int

__all__ = []


@dispatch(NPNumeric)
def isnan(a):
    return np.isnan(a)


@dispatch(NPDType, [Int])
def zeros(dtype, *shape):
    return np.zeros(shape, dtype=dtype)


@dispatch(NPDType, [Int])
def ones(dtype, *shape):
    return np.ones(shape, dtype=dtype)


@dispatch(NPDType, Int, Int)
def eye(dtype, *shape):
    return np.eye(shape[0], shape[1], dtype=dtype)


@dispatch(NPDType, object, object, Int)
def linspace(dtype, a, b, num):
    return np.linspace(a, b, num, dtype=dtype)


@dispatch(NPDType, object, object, object)
def range(dtype, start, stop, step):
    return np.arange(start, stop, step, dtype=dtype)


@dispatch(NPDType, NPNumeric)
def cast(dtype, a):
    if hasattr(a, 'astype'):
        return a.astype(dtype, copy=False)
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


dispatch(NPNumeric, NPNumeric, NPNumeric)(bvn_cdf)


@dispatch(NPNumeric)
def sort(a, axis=-1, descending=False):
    if descending:
        return -np.sort(-a, axis=axis)
    else:
        return np.sort(a, axis=axis)


@dispatch(NPNumeric)
def argsort(a, axis=-1, descending=False):
    if descending:
        return np.argsort(-a, axis=axis)
    else:
        return np.argsort(a, axis=axis)
