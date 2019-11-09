import tensorflow as tf
from plum import Callable

from . import dispatch, B, Numeric, TFNumeric
from .custom import tensorflow_register
from ..custom import bvn_cdf, s_bvn_cdf
from ..types import TFDType, Int

__all__ = []


@dispatch(Numeric)
def isnan(a):
    return tf.math.is_nan(a)


@dispatch(TFDType, [Int])
def zeros(dtype, *shape):
    return tf.zeros(shape, dtype=dtype)


@dispatch(TFDType, [Int])
def ones(dtype, *shape):
    return tf.ones(shape, dtype=dtype)


@dispatch(TFDType, Int, Int)
def eye(dtype, *shape):
    return tf.eye(shape[0], shape[1], dtype=dtype)


@dispatch(TFDType, object, object, Int)
def linspace(dtype, a, b, num):
    return tf.linspace(cast(dtype, a), cast(dtype, b), num)


@dispatch(TFDType, object, object, object)
def range(dtype, start, stop, step):
    return tf.range(start, stop, step, dtype=dtype)


@dispatch(TFDType, Numeric)
def cast(dtype, a):
    return tf.cast(a, dtype=dtype)


@dispatch(Numeric)
def identity(a):
    return tf.identity(a)


@dispatch(Numeric)
def negative(a):
    return tf.negative(a)


@dispatch(Numeric)
def abs(a):
    return tf.abs(a)


@dispatch(Numeric)
def sign(a):
    return tf.sign(a)


@dispatch(Numeric)
def sqrt(a):
    return tf.sqrt(a)


@dispatch(Numeric)
def exp(a):
    return tf.exp(a)


@dispatch(Numeric)
def log(a):
    return tf.math.log(a)


@dispatch(Numeric)
def sin(a):
    return tf.sin(a)


@dispatch(Numeric)
def cos(a):
    return tf.cos(a)


@dispatch(Numeric)
def tan(a):
    return tf.tan(a)


@dispatch(Numeric)
def tanh(a):
    return tf.tanh(a)


@dispatch(Numeric, Numeric)
def add(a, b):
    return tf.add(a, b)


@dispatch(Numeric, Numeric)
def subtract(a, b):
    return tf.subtract(a, b)


@dispatch(Numeric, Numeric)
def multiply(a, b):
    return tf.multiply(a, b)


@dispatch(Numeric, Numeric)
def divide(a, b):
    return tf.divide(a, b)


@dispatch(Numeric, Numeric)
def power(a, b):
    return tf.pow(a, b)


@dispatch(Numeric, Numeric)
def minimum(a, b):
    return tf.minimum(a, b)


@dispatch(Numeric, Numeric)
def maximum(a, b):
    return tf.maximum(a, b)


@dispatch(Numeric)
def min(a, axis=None):
    return tf.reduce_min(a, axis=axis)


@dispatch(Numeric)
def max(a, axis=None):
    return tf.reduce_max(a, axis=axis)


@dispatch(Numeric)
def sum(a, axis=None):
    return tf.reduce_sum(a, axis=axis)


@dispatch(Numeric)
def mean(a, axis=None):
    return tf.reduce_mean(a, axis=axis)


@dispatch(Numeric)
def std(a, axis=None):
    if axis is None:
        axes = list(range(B.rank(a)))
    else:
        axes = [axis]
    _, var = tf.nn.moments(a, axes=axes)
    return tf.sqrt(var)


@dispatch(Numeric)
def all(a, axis=None):
    return tf.reduce_all(a, axis=axis)


@dispatch(Numeric)
def any(a, axis=None):
    return tf.reduce_any(a, axis=axis)


@dispatch(Numeric, Numeric)
def lt(a, b):
    return tf.less(a, b)


@dispatch(Numeric, Numeric)
def le(a, b):
    return tf.less_equal(a, b)


@dispatch(Numeric, Numeric)
def gt(a, b):
    return tf.greater(a, b)


@dispatch(Numeric, Numeric)
def ge(a, b):
    return tf.greater_equal(a, b)


f = tensorflow_register(bvn_cdf, s_bvn_cdf)
dispatch(Numeric, Numeric, Numeric)(f)


# If `Numeric` types are used here, this implementation is more specific
# than the generic implementation, which will use TensorFlow unnecessarily.
@dispatch(Callable, TFNumeric, [TFNumeric])
def scan(f, xs, *init_state):
    return tf.scan(f, xs, initializer=init_state)


@dispatch(Numeric)
def sort(a, axis=-1, descending=False):
    if descending:
        direction = 'DESCENDING'
    else:
        direction = 'ASCENDING'
    return tf.sort(a, axis=axis, direction=direction)


@dispatch(Numeric)
def argsort(a, axis=-1, descending=False):
    if descending:
        direction = 'DESCENDING'
    else:
        direction = 'ASCENDING'
    return tf.argsort(a, axis=axis, direction=direction)
