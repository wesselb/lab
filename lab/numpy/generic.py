import numpy as np
import scipy.special as sps

from . import B, dispatch, Numeric
from ..custom import bvn_cdf as _bvn_cdf
from ..types import NPNumeric, NPDType, Int

__all__ = []


@dispatch
def isnan(a: Numeric):
    return np.isnan(a)


@dispatch
def device(a: NPNumeric):
    return "cpu"


@dispatch
def move_to_active_device(a: NPNumeric):
    return a


@dispatch
def zeros(dtype: NPDType, *shape: Int):
    return np.zeros(shape, dtype=dtype)


@dispatch
def ones(dtype: NPDType, *shape: Int):
    return np.ones(shape, dtype=dtype)


@dispatch
def _eye2(dtype: NPDType, *shape: Int):
    return np.eye(shape[0], shape[1], dtype=dtype)


@dispatch
def linspace(dtype: NPDType, a, b, num: Int):
    return np.linspace(a, b, num, dtype=dtype)


@dispatch
def range(dtype: NPDType, start, stop, step):
    return np.arange(start, stop, step, dtype=dtype)


@dispatch
def cast(dtype: NPDType, a: Numeric):
    if B.dtype(a) == dtype:
        return a
    if hasattr(a, "astype"):
        return a.astype(dtype, copy=False)
    else:
        return np.array(a, dtype=dtype)


@dispatch
def identity(a: Numeric):
    return np.array(a)


@dispatch
def negative(a: Numeric):
    return np.negative(a)


@dispatch
def abs(a: Numeric):
    return np.abs(a)


@dispatch
def sign(a: Numeric):
    return np.sign(a)


@dispatch
def sqrt(a: Numeric):
    return np.sqrt(a)


@dispatch
def exp(a: Numeric):
    return np.exp(a)


@dispatch
def log(a: Numeric):
    return np.log(a)


@dispatch
def sin(a: Numeric):
    return np.sin(a)


@dispatch
def cos(a: Numeric):
    return np.cos(a)


@dispatch
def tan(a: Numeric):
    return np.tan(a)


@dispatch
def tanh(a: Numeric):
    return np.tanh(a)


@dispatch
def erf(a: Numeric):
    return sps.erf(a)


@dispatch
def add(a: Numeric, b: Numeric):
    return np.add(a, b)


@dispatch
def subtract(a: Numeric, b: Numeric):
    return np.subtract(a, b)


@dispatch
def multiply(a: Numeric, b: Numeric):
    return np.multiply(a, b)


@dispatch
def divide(a: Numeric, b: Numeric):
    return np.divide(a, b)


@dispatch
def power(a: Numeric, b: Numeric):
    return np.power(a, b)


@dispatch
def minimum(a: Numeric, b: Numeric):
    return np.minimum(a, b)


@dispatch
def maximum(a: Numeric, b: Numeric):
    return np.maximum(a, b)


@dispatch
def min(a: Numeric, axis=None):
    return np.min(a, axis=axis)


@dispatch
def max(a: Numeric, axis=None):
    return np.max(a, axis=axis)


@dispatch
def sum(a: Numeric, axis=None):
    return np.sum(a, axis=axis)


@dispatch
def mean(a: Numeric, axis=None):
    return np.mean(a, axis=axis)


@dispatch
def std(a: Numeric, axis=None):
    return np.std(a, axis=axis, ddof=0)


@dispatch
def all(a: Numeric, axis=None):
    return np.all(a, axis=axis)


@dispatch
def any(a: Numeric, axis=None):
    return np.any(a, axis=axis)


@dispatch
def lt(a: Numeric, b: Numeric):
    return np.less(a, b)


@dispatch
def le(a: Numeric, b: Numeric):
    return np.less_equal(a, b)


@dispatch
def gt(a: Numeric, b: Numeric):
    return np.greater(a, b)


@dispatch
def ge(a: Numeric, b: Numeric):
    return np.greater_equal(a, b)


@dispatch
def bvn_cdf(a: Numeric, b: Numeric, c: Numeric):
    return _bvn_cdf(a, b, c)


@dispatch
def sort(a: Numeric, axis=-1, descending=False):
    if descending:
        return -np.sort(-a, axis=axis)
    else:
        return np.sort(a, axis=axis)


@dispatch
def argsort(a: Numeric, axis=-1, descending=False):
    if descending:
        return np.argsort(-a, axis=axis)
    else:
        return np.argsort(a, axis=axis)
