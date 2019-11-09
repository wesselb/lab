import autograd.numpy as anp

from . import dispatch, Numeric
from .custom import autograd_register
from ..custom import bvn_cdf, s_bvn_cdf
from ..types import AGDType, AGNumeric

__all__ = []


@dispatch(Numeric)
def isnan(a):
    return anp.isnan(a)


@dispatch(AGDType, AGNumeric)
def cast(dtype, a):
    # AutoGrad does not respect the `copy` flag, so check that manually.
    if dtype == a.dtype:
        return a
    else:
        return a.astype(dtype)


@dispatch(Numeric)
def identity(a):
    return anp.array(a)


@dispatch(Numeric)
def negative(a):
    return anp.negative(a)


@dispatch(Numeric)
def abs(a):
    return anp.abs(a)


@dispatch(Numeric)
def sign(a):
    return anp.sign(a)


@dispatch(Numeric)
def sqrt(a):
    return anp.sqrt(a)


@dispatch(Numeric)
def exp(a):
    return anp.exp(a)


@dispatch(Numeric)
def log(a):
    return anp.log(a)


@dispatch(Numeric)
def sin(a):
    return anp.sin(a)


@dispatch(Numeric)
def cos(a):
    return anp.cos(a)


@dispatch(Numeric)
def tan(a):
    return anp.tan(a)


@dispatch(Numeric)
def tanh(a):
    return anp.tanh(a)


@dispatch(Numeric, Numeric)
def add(a, b):
    return anp.add(a, b)


@dispatch(Numeric, Numeric)
def subtract(a, b):
    return anp.subtract(a, b)


@dispatch(Numeric, Numeric)
def multiply(a, b):
    return anp.multiply(a, b)


@dispatch(Numeric, Numeric)
def divide(a, b):
    return anp.divide(a, b)


@dispatch(Numeric, Numeric)
def power(a, b):
    return anp.power(a, b)


@dispatch(Numeric, Numeric)
def minimum(a, b):
    return anp.minimum(a, b)


@dispatch(Numeric, Numeric)
def maximum(a, b):
    return anp.maximum(a, b)


@dispatch(Numeric)
def min(a, axis=None):
    return anp.min(a, axis=axis)


@dispatch(Numeric)
def max(a, axis=None):
    return anp.max(a, axis=axis)


@dispatch(Numeric)
def sum(a, axis=None):
    return anp.sum(a, axis=axis)


@dispatch(Numeric)
def mean(a, axis=None):
    return anp.mean(a, axis=axis)


@dispatch(Numeric)
def std(a, axis=None):
    return anp.std(a, axis=axis, ddof=0)


@dispatch(Numeric)
def all(a, axis=None):
    return anp.all(a, axis=axis)


@dispatch(Numeric)
def any(a, axis=None):
    return anp.any(a, axis=axis)


@dispatch(Numeric, Numeric)
def lt(a, b):
    return anp.less(a, b)


@dispatch(Numeric, Numeric)
def le(a, b):
    return anp.less_equal(a, b)


@dispatch(Numeric, Numeric)
def gt(a, b):
    return anp.greater(a, b)


@dispatch(Numeric, Numeric)
def ge(a, b):
    return anp.greater_equal(a, b)


f = autograd_register(bvn_cdf, s_bvn_cdf)
dispatch(Numeric, Numeric, Numeric)(f)


@dispatch(Numeric)
def sort(a, axis=-1, descending=False):
    if descending:
        return -anp.sort(-a, axis=axis)
    else:
        return anp.sort(a, axis=axis)


@dispatch(Numeric)
def argsort(a, axis=-1, descending=False):
    if descending:
        return anp.argsort(-a, axis=axis)
    else:
        return anp.argsort(a, axis=axis)
