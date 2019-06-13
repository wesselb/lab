# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from plum import convert, Callable

from . import dispatch, B
from .custom import tensorflow_register
from ..custom import bvn_cdf, s_bvn_cdf
from ..types import TFNumeric, TFDType, NPNumeric, Number, Int

__all__ = []


@dispatch(TFNumeric)
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


@dispatch(TFDType, {TFNumeric, NPNumeric, Number})
def cast(dtype, a):
    return tf.cast(a, dtype=dtype)


@dispatch(TFNumeric)
def identity(a):
    return tf.identity(a)


@dispatch(TFNumeric)
def abs(a):
    return tf.abs(a)


@dispatch(TFNumeric)
def sign(a):
    return tf.sign(a)


@dispatch(TFNumeric)
def sqrt(a):
    return tf.sqrt(a)


@dispatch(TFNumeric)
def exp(a):
    return tf.exp(a)


@dispatch(TFNumeric)
def log(a):
    return tf.math.log(a)


@dispatch(TFNumeric)
def sin(a):
    return tf.sin(a)


@dispatch(TFNumeric)
def cos(a):
    return tf.cos(a)


@dispatch(TFNumeric)
def tan(a):
    return tf.tan(a)


@dispatch(TFNumeric)
def tanh(a):
    return tf.tanh(a)


@dispatch(TFNumeric, TFNumeric)
def add(a, b):
    return tf.add(a, b)


@dispatch(TFNumeric, TFNumeric)
def subtract(a, b):
    return tf.subtract(a, b)


@dispatch(TFNumeric, TFNumeric)
def multiply(a, b):
    return tf.multiply(a, b)


@dispatch(TFNumeric, TFNumeric)
def divide(a, b):
    return tf.divide(a, b)


@dispatch(TFNumeric, TFNumeric)
def power(a, b):
    return tf.pow(a, b)


@dispatch(TFNumeric, TFNumeric)
def minimum(a, b):
    return tf.minimum(a, b)


@dispatch(TFNumeric, TFNumeric)
def maximum(a, b):
    return tf.maximum(a, b)


@dispatch(TFNumeric)
def min(a, axis=None):
    return tf.reduce_min(a, axis=axis)


@dispatch(TFNumeric)
def max(a, axis=None):
    return tf.reduce_max(a, axis=axis)


@dispatch(TFNumeric)
def sum(a, axis=None):
    return tf.reduce_sum(a, axis=axis)


@dispatch(TFNumeric)
def mean(a, axis=None):
    return tf.reduce_mean(a, axis=axis)


@dispatch(TFNumeric)
def std(a, axis=None):
    if axis is None:
        axes = list(range(B.rank(a)))
    else:
        axes = [axis]
    _, var = tf.nn.moments(a, axes=axes)
    return tf.sqrt(var)


@dispatch(TFNumeric)
def all(a, axis=None):
    return tf.reduce_all(a, axis=axis)


@dispatch(TFNumeric)
def any(a, axis=None):
    return tf.reduce_any(a, axis=axis)


@dispatch(TFNumeric, TFNumeric)
def lt(a, b):
    return tf.less(a, b)


@dispatch(TFNumeric, TFNumeric)
def le(a, b):
    return tf.less_equal(a, b)


@dispatch(TFNumeric, TFNumeric)
def gt(a, b):
    return tf.greater(a, b)


@dispatch(TFNumeric, TFNumeric)
def ge(a, b):
    return tf.greater_equal(a, b)


f = tensorflow_register(bvn_cdf, s_bvn_cdf)
dispatch(TFNumeric, TFNumeric, TFNumeric)(f)


@dispatch(Callable, TFNumeric, [TFNumeric])
def scan(f, xs, *init_state):
    return tf.scan(f, xs, initializer=init_state)


@dispatch(TFNumeric)
def sort(a, axis=-1, descending=False):
    if descending:
        direction = 'DESCENDING'
    else:
        direction = 'ASCENDING'
    return tf.sort(a, axis=axis, direction=direction)


@dispatch(TFNumeric)
def argsort(a, axis=-1, descending=False):
    if descending:
        direction = 'DESCENDING'
    else:
        direction = 'ASCENDING'
    return tf.argsort(a, axis=axis, direction=direction)
