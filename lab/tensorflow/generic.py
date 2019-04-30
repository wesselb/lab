# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from . import dispatch, B
from ..types import TFNumeric, TFDType, TFShape

__all__ = []


@dispatch(TFShape, TFDType)
def zeros(shape, dtype):
    return tf.zeros(shape, dtype=dtype)


@dispatch(TFShape, TFDType)
def ones(shape, dtype):
    return tf.ones(shape, dtype=dtype)


@dispatch(TFNumeric, TFDType)
def cast(a, dtype):
    return tf.cast(a, dtype=dtype)


@dispatch(TFNumeric)
def abs(a):
    return tf.abs(a)


@dispatch(TFNumeric)
def exp(a):
    return tf.exp(a)


@dispatch(TFNumeric)
def log(a):
    return tf.log(a)


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
