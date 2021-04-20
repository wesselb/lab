from types import FunctionType
from typing import Callable

import tensorflow as tf

from . import dispatch, B, Numeric, TFNumeric
from .custom import tensorflow_register
from ..custom import bvn_cdf, s_bvn_cdf
from ..types import TFDType, Int

__all__ = []


@dispatch
def isnan(a: Numeric):
    return tf.math.is_nan(a)


@dispatch
def device(a: TFNumeric):
    return str(a.device)


@dispatch
def move_to_active_device(a: TFNumeric):
    return a


@dispatch
def zeros(dtype: TFDType, *shape: Int):
    return tf.zeros(shape, dtype=dtype)


@dispatch
def ones(dtype: TFDType, *shape: Int):
    return tf.ones(shape, dtype=dtype)


@dispatch
def _eye2(dtype: TFDType, *shape: Int):
    return tf.eye(shape[0], shape[1], dtype=dtype)


@dispatch
def linspace(dtype: TFDType, a, b, num: Int):
    return tf.linspace(cast(dtype, a), cast(dtype, b), num)


@dispatch
def range(dtype: TFDType, start, stop, step):
    return tf.range(start, stop, step, dtype=dtype)


@dispatch
def cast(dtype: TFDType, a: Numeric):
    return tf.cast(a, dtype=dtype)


@dispatch
def identity(a: Numeric):
    return tf.identity(a)


@dispatch
def negative(a: Numeric):
    return tf.negative(a)


@dispatch
def abs(a: Numeric):
    return tf.abs(a)


@dispatch
def sign(a: Numeric):
    return tf.sign(a)


@dispatch
def sqrt(a: Numeric):
    return tf.sqrt(a)


@dispatch
def exp(a: Numeric):
    return tf.exp(a)


@dispatch
def log(a: Numeric):
    return tf.math.log(a)


@dispatch
def sin(a: Numeric):
    return tf.sin(a)


@dispatch
def cos(a: Numeric):
    return tf.cos(a)


@dispatch
def tan(a: Numeric):
    return tf.tan(a)


@dispatch
def tanh(a: Numeric):
    return tf.tanh(a)


@dispatch
def erf(a: Numeric):
    return tf.math.erf(a)


@dispatch
def add(a: Numeric, b: Numeric):
    return tf.add(a, b)


@dispatch
def subtract(a: Numeric, b: Numeric):
    return tf.subtract(a, b)


@dispatch
def multiply(a: Numeric, b: Numeric):
    return tf.multiply(a, b)


@dispatch
def divide(a: Numeric, b: Numeric):
    return tf.divide(a, b)


@dispatch
def power(a: Numeric, b: Numeric):
    return tf.pow(a, b)


@dispatch
def minimum(a: Numeric, b: Numeric):
    return tf.minimum(a, b)


@dispatch
def maximum(a: Numeric, b: Numeric):
    return tf.maximum(a, b)


@dispatch
def min(a: Numeric, axis=None):
    return tf.reduce_min(a, axis=axis)


@dispatch
def max(a: Numeric, axis=None):
    return tf.reduce_max(a, axis=axis)


@dispatch
def sum(a: Numeric, axis=None):
    return tf.reduce_sum(a, axis=axis)


@dispatch
def mean(a: Numeric, axis=None):
    return tf.reduce_mean(a, axis=axis)


@dispatch
def std(a: Numeric, axis=None):
    if axis is None:
        axes = list(range(B.rank(a)))
    else:
        axes = [axis]
    _, var = tf.nn.moments(a, axes=axes)
    return tf.sqrt(var)


@dispatch
def all(a: Numeric, axis=None):
    return tf.reduce_all(a, axis=axis)


@dispatch
def any(a: Numeric, axis=None):
    return tf.reduce_any(a, axis=axis)


@dispatch
def lt(a: Numeric, b: Numeric):
    return tf.less(a, b)


@dispatch
def le(a: Numeric, b: Numeric):
    return tf.less_equal(a, b)


@dispatch
def gt(a: Numeric, b: Numeric):
    return tf.greater(a, b)


@dispatch
def ge(a: Numeric, b: Numeric):
    return tf.greater_equal(a, b)


_bvn_cdf = tensorflow_register(bvn_cdf, s_bvn_cdf)


@dispatch
def bvn_cdf(a: Numeric, b: Numeric, c: Numeric):
    return _bvn_cdf(a, b, c)


@dispatch
def _cond(
    condition: Numeric, f_true: FunctionType, f_false: FunctionType, *xs: TFNumeric
):
    return tf.cond(condition, lambda: f_true(*xs), lambda: f_false(*xs))


# If `Numeric` types are used here, this implementation is more specific than the
# generic implementation, which will use TensorFlow unnecessarily.
@dispatch
def scan(f: Callable, xs: TFNumeric, *init_state: TFNumeric):
    return tf.scan(f, xs, initializer=init_state)


@dispatch
def sort(a: Numeric, axis=-1, descending=False):
    if descending:
        direction = "DESCENDING"
    else:
        direction = "ASCENDING"
    return tf.sort(a, axis=axis, direction=direction)


@dispatch
def argsort(a: Numeric, axis=-1, descending=False):
    if descending:
        direction = "DESCENDING"
    else:
        direction = "ASCENDING"
    return tf.argsort(a, axis=axis, direction=direction)
