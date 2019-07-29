# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import autograd.numpy as anp

from . import dispatch
from .custom import autograd_register
from ..custom import bvn_cdf, s_bvn_cdf
from ..types import AGNumeric, AGDType

__all__ = []


@dispatch(AGNumeric)
def isnan(a):
    return anp.isnan(a)


@dispatch(AGDType, AGNumeric)
def cast(dtype, a):
    if hasattr(a, 'astype'):
        return a.astype(dtype, copy=True)
    else:
        return anp.array(a, dtype=dtype)


@dispatch(AGNumeric)
def identity(a):
    return anp.array(a)


@dispatch(AGNumeric)
def abs(a):
    return anp.abs(a)


@dispatch(AGNumeric)
def sign(a):
    return anp.sign(a)


@dispatch(AGNumeric)
def sqrt(a):
    return anp.sqrt(a)


@dispatch(AGNumeric)
def exp(a):
    return anp.exp(a)


@dispatch(AGNumeric)
def log(a):
    return anp.log(a)


@dispatch(AGNumeric)
def sin(a):
    return anp.sin(a)


@dispatch(AGNumeric)
def cos(a):
    return anp.cos(a)


@dispatch(AGNumeric)
def tan(a):
    return anp.tan(a)


@dispatch(AGNumeric)
def tanh(a):
    return anp.tanh(a)


@dispatch(AGNumeric, AGNumeric)
def add(a, b):
    return anp.add(a, b)


@dispatch(AGNumeric, AGNumeric)
def subtract(a, b):
    return anp.subtract(a, b)


@dispatch(AGNumeric, AGNumeric)
def multiply(a, b):
    return anp.multiply(a, b)


@dispatch(AGNumeric, AGNumeric)
def divide(a, b):
    return anp.divide(a, b)


@dispatch(AGNumeric, AGNumeric)
def power(a, b):
    return anp.power(a, b)


@dispatch(AGNumeric, AGNumeric)
def minimum(a, b):
    return anp.minimum(a, b)


@dispatch(AGNumeric, AGNumeric)
def maximum(a, b):
    return anp.maximum(a, b)


@dispatch(AGNumeric)
def min(a, axis=None):
    return anp.min(a, axis=axis)


@dispatch(AGNumeric)
def max(a, axis=None):
    return anp.max(a, axis=axis)


@dispatch(AGNumeric)
def sum(a, axis=None):
    return anp.sum(a, axis=axis)


@dispatch(AGNumeric)
def mean(a, axis=None):
    return anp.mean(a, axis=axis)


@dispatch(AGNumeric)
def std(a, axis=None):
    return anp.std(a, axis=axis, ddof=0)


@dispatch(AGNumeric)
def all(a, axis=None):
    return anp.all(a, axis=axis)


@dispatch(AGNumeric)
def any(a, axis=None):
    return anp.any(a, axis=axis)


@dispatch(AGNumeric, AGNumeric)
def lt(a, b):
    return anp.less(a, b)


@dispatch(AGNumeric, AGNumeric)
def le(a, b):
    return anp.less_equal(a, b)


@dispatch(AGNumeric, AGNumeric)
def gt(a, b):
    return anp.greater(a, b)


@dispatch(AGNumeric, AGNumeric)
def ge(a, b):
    return anp.greater_equal(a, b)


f = autograd_register(bvn_cdf, s_bvn_cdf)
dispatch(AGNumeric, AGNumeric, AGNumeric)(f)


@dispatch(AGNumeric)
def sort(a, axis=-1, descending=False):
    if descending:
        return -anp.sort(-a, axis=axis)
    else:
        return anp.sort(a, axis=axis)


@dispatch(AGNumeric)
def argsort(a, axis=-1, descending=False):
    if descending:
        return anp.argsort(-a, axis=axis)
    else:
        return anp.argsort(a, axis=axis)
