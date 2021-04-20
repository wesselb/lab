import autograd.numpy as anp
import autograd.scipy.special as asps

from . import dispatch, Numeric
from .custom import autograd_register
from ..custom import bvn_cdf, s_bvn_cdf
from ..types import AGDType, AGNumeric

__all__ = []


@dispatch
def isnan(a: Numeric):
    return anp.isnan(a)


@dispatch
def device(a: AGNumeric):
    return "cpu"


@dispatch
def move_to_active_device(a: AGNumeric):
    return a


@dispatch
def cast(dtype: AGDType, a: AGNumeric):
    # AutoGrad does not respect the `copy` flag, so check that manually.
    if dtype == a.dtype:
        return a
    else:
        return a.astype(dtype)


@dispatch
def identity(a: Numeric):
    return 1 * a


@dispatch
def negative(a: Numeric):
    return anp.negative(a)


@dispatch
def abs(a: Numeric):
    return anp.abs(a)


@dispatch
def sign(a: Numeric):
    return anp.sign(a)


@dispatch
def sqrt(a: Numeric):
    return anp.sqrt(a)


@dispatch
def exp(a: Numeric):
    return anp.exp(a)


@dispatch
def log(a: Numeric):
    return anp.log(a)


@dispatch
def sin(a: Numeric):
    return anp.sin(a)


@dispatch
def cos(a: Numeric):
    return anp.cos(a)


@dispatch
def tan(a: Numeric):
    return anp.tan(a)


@dispatch
def tanh(a: Numeric):
    return anp.tanh(a)


@dispatch
def erf(a: Numeric):
    return asps.erf(a)


@dispatch
def add(a: Numeric, b: Numeric):
    return anp.add(a, b)


@dispatch
def subtract(a: Numeric, b: Numeric):
    return anp.subtract(a, b)


@dispatch
def multiply(a: Numeric, b: Numeric):
    return anp.multiply(a, b)


@dispatch
def divide(a: Numeric, b: Numeric):
    return anp.divide(a, b)


@dispatch
def power(a: Numeric, b: Numeric):
    return anp.power(a, b)


@dispatch
def minimum(a: Numeric, b: Numeric):
    return anp.minimum(a, b)


@dispatch
def maximum(a: Numeric, b: Numeric):
    return anp.maximum(a, b)


@dispatch
def min(a: Numeric, axis=None):
    return anp.min(a, axis=axis)


@dispatch
def max(a: Numeric, axis=None):
    return anp.max(a, axis=axis)


@dispatch
def sum(a: Numeric, axis=None):
    return anp.sum(a, axis=axis)


@dispatch
def mean(a: Numeric, axis=None):
    return anp.mean(a, axis=axis)


@dispatch
def std(a: Numeric, axis=None):
    return anp.std(a, axis=axis, ddof=0)


@dispatch
def all(a: Numeric, axis=None):
    return anp.all(a, axis=axis)


@dispatch
def any(a: Numeric, axis=None):
    return anp.any(a, axis=axis)


@dispatch
def lt(a: Numeric, b: Numeric):
    return anp.less(a, b)


@dispatch
def le(a: Numeric, b: Numeric):
    return anp.less_equal(a, b)


@dispatch
def gt(a: Numeric, b: Numeric):
    return anp.greater(a, b)


@dispatch
def ge(a: Numeric, b: Numeric):
    return anp.greater_equal(a, b)


_bvn_cdf = autograd_register(bvn_cdf, s_bvn_cdf)


@dispatch
def bvn_cdf(a: Numeric, b: Numeric, c: Numeric):
    return _bvn_cdf(a, b, c)


@dispatch
def sort(a: Numeric, axis=-1, descending=False):
    if descending:
        return -anp.sort(-a, axis=axis)
    else:
        return anp.sort(a, axis=axis)


@dispatch
def argsort(a: Numeric, axis=-1, descending=False):
    if descending:
        return anp.argsort(-a, axis=axis)
    else:
        return anp.argsort(a, axis=axis)
