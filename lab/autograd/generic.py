# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import autograd.numpy as anp

from . import dispatch
from .custom import autograd_register
from ..custom import bvn_cdf, s_bvn_cdf
from ..types import NPNumeric, NPDType, Int

__all__ = []


@dispatch(NPNumeric)
def isnan(a):
    return anp.isnan(a)


@dispatch(NPDType, [Int])
def zeros(dtype, *shape):
    return anp.zeros(shape, dtype=dtype)


@dispatch(NPDType, [Int])
def ones(dtype, *shape):
    return anp.ones(shape, dtype=dtype)


@dispatch(NPDType, Int, Int)
def eye(dtype, *shape):
    return anp.eye(shape[0], shape[1], dtype=dtype)


@dispatch(NPDType, object, object, Int)
def linspace(dtype, a, b, num):
    return anp.linspace(a, b, num, dtype=dtype)


@dispatch(NPDType, object, object, object)
def range(dtype, start, stop, step):
    return anp.arange(start, stop, step, dtype=dtype)


@dispatch(NPDType, NPNumeric)
def cast(dtype, a):
    if hasattr(a, 'astype'):
        return a.astype(dtype, copy=False)
    else:
        return anp.array(a, dtype=dtype)


@dispatch(NPNumeric)
def identity(a):
    return anp.array(a)


@dispatch(NPNumeric)
def abs(a):
    return anp.abs(a)


@dispatch(NPNumeric)
def sign(a):
    return anp.sign(a)


@dispatch(NPNumeric)
def sqrt(a):
    return anp.sqrt(a)


@dispatch(NPNumeric)
def exp(a):
    return anp.exp(a)


@dispatch(NPNumeric)
def log(a):
    return anp.log(a)


@dispatch(NPNumeric)
def sin(a):
    return anp.sin(a)


@dispatch(NPNumeric)
def cos(a):
    return anp.cos(a)


@dispatch(NPNumeric)
def tan(a):
    return anp.tan(a)


@dispatch(NPNumeric)
def tanh(a):
    return anp.tanh(a)


@dispatch(NPNumeric, NPNumeric)
def add(a, b):
    return anp.add(a, b)


@dispatch(NPNumeric, NPNumeric)
def subtract(a, b):
    return anp.subtract(a, b)


@dispatch(NPNumeric, NPNumeric)
def multiply(a, b):
    return anp.multiply(a, b)


@dispatch(NPNumeric, NPNumeric)
def divide(a, b):
    return anp.divide(a, b)


@dispatch(NPNumeric, NPNumeric)
def power(a, b):
    return anp.power(a, b)


@dispatch(NPNumeric, NPNumeric)
def minimum(a, b):
    return anp.minimum(a, b)


@dispatch(NPNumeric, NPNumeric)
def maximum(a, b):
    return anp.maximum(a, b)


@dispatch(NPNumeric)
def min(a, axis=None):
    return anp.min(a, axis=axis)


@dispatch(NPNumeric)
def max(a, axis=None):
    return anp.max(a, axis=axis)


@dispatch(NPNumeric)
def sum(a, axis=None):
    return anp.sum(a, axis=axis)


@dispatch(NPNumeric)
def mean(a, axis=None):
    return anp.mean(a, axis=axis)


@dispatch(NPNumeric)
def std(a, axis=None):
    return anp.std(a, axis=axis, ddof=0)


@dispatch(NPNumeric)
def all(a, axis=None):
    return anp.all(a, axis=axis)


@dispatch(NPNumeric)
def any(a, axis=None):
    return anp.any(a, axis=axis)


@dispatch(NPNumeric, NPNumeric)
def lt(a, b):
    return anp.less(a, b)


@dispatch(NPNumeric, NPNumeric)
def le(a, b):
    return anp.less_equal(a, b)


@dispatch(NPNumeric, NPNumeric)
def gt(a, b):
    return anp.greater(a, b)


@dispatch(NPNumeric, NPNumeric)
def ge(a, b):
    return anp.greater_equal(a, b)


f = autograd_register(bvn_cdf, s_bvn_cdf)
dispatch(NPNumeric, NPNumeric, NPNumeric)(f)


@dispatch(NPNumeric)
def sort(a, axis=-1, descending=False):
    if descending:
        return -anp.sort(-a, axis=axis)
    else:
        return anp.sort(a, axis=axis)


@dispatch(NPNumeric)
def argsort(a, axis=-1, descending=False):
    if descending:
        return anp.argsort(-a, axis=axis)
    else:
        return anp.argsort(a, axis=axis)
