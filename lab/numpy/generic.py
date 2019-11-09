import numpy as np

from . import dispatch, Numeric
from ..custom import bvn_cdf
from ..types import NPDType, Int

__all__ = []


@dispatch(Numeric)
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


@dispatch(NPDType, Numeric)
def cast(dtype, a):
    if hasattr(a, 'astype'):
        return a.astype(dtype, copy=False)
    else:
        return np.array(a, dtype=dtype)


@dispatch(Numeric)
def identity(a):
    return np.array(a)


@dispatch(Numeric)
def negative(a):
    return np.negative(a)


@dispatch(Numeric)
def abs(a):
    return np.abs(a)


@dispatch(Numeric)
def sign(a):
    return np.sign(a)


@dispatch(Numeric)
def sqrt(a):
    return np.sqrt(a)


@dispatch(Numeric)
def exp(a):
    return np.exp(a)


@dispatch(Numeric)
def log(a):
    return np.log(a)


@dispatch(Numeric)
def sin(a):
    return np.sin(a)


@dispatch(Numeric)
def cos(a):
    return np.cos(a)


@dispatch(Numeric)
def tan(a):
    return np.tan(a)


@dispatch(Numeric)
def tanh(a):
    return np.tanh(a)


@dispatch(Numeric, Numeric)
def add(a, b):
    return np.add(a, b)


@dispatch(Numeric, Numeric)
def subtract(a, b):
    return np.subtract(a, b)


@dispatch(Numeric, Numeric)
def multiply(a, b):
    return np.multiply(a, b)


@dispatch(Numeric, Numeric)
def divide(a, b):
    return np.divide(a, b)


@dispatch(Numeric, Numeric)
def power(a, b):
    return np.power(a, b)


@dispatch(Numeric, Numeric)
def minimum(a, b):
    return np.minimum(a, b)


@dispatch(Numeric, Numeric)
def maximum(a, b):
    return np.maximum(a, b)


@dispatch(Numeric)
def min(a, axis=None):
    return np.min(a, axis=axis)


@dispatch(Numeric)
def max(a, axis=None):
    return np.max(a, axis=axis)


@dispatch(Numeric)
def sum(a, axis=None):
    return np.sum(a, axis=axis)


@dispatch(Numeric)
def mean(a, axis=None):
    return np.mean(a, axis=axis)


@dispatch(Numeric)
def std(a, axis=None):
    return np.std(a, axis=axis, ddof=0)


@dispatch(Numeric)
def all(a, axis=None):
    return np.all(a, axis=axis)


@dispatch(Numeric)
def any(a, axis=None):
    return np.any(a, axis=axis)


@dispatch(Numeric, Numeric)
def lt(a, b):
    return np.less(a, b)


@dispatch(Numeric, Numeric)
def le(a, b):
    return np.less_equal(a, b)


@dispatch(Numeric, Numeric)
def gt(a, b):
    return np.greater(a, b)


@dispatch(Numeric, Numeric)
def ge(a, b):
    return np.greater_equal(a, b)


dispatch(Numeric, Numeric, Numeric)(bvn_cdf)


@dispatch(Numeric)
def sort(a, axis=-1, descending=False):
    if descending:
        return -np.sort(-a, axis=axis)
    else:
        return np.sort(a, axis=axis)


@dispatch(Numeric)
def argsort(a, axis=-1, descending=False):
    if descending:
        return np.argsort(-a, axis=axis)
    else:
        return np.argsort(a, axis=axis)
