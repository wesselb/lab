# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
import tensorflow as tf

from . import dispatch
from ..types import TFDimension, TFDType, TFNumeric, Int
from plum import convert

__all__ = []
log = logging.getLogger(__name__)


@dispatch(TFDType, [Int])
def rand(dtype, *shape):
    return tf.random_uniform(shape, dtype=dtype)


@dispatch(TFDType, [TFDimension])
def rand(dtype, *shape):
    # `tf.random_normal` requires integers!
    return rand(dtype, *[convert(s, int) for s in shape])


@dispatch(TFDType, [Int])
def randn(dtype, *shape):
    return tf.random_normal(shape, dtype=dtype)


@dispatch(TFDType, [TFDimension])
def randn(dtype, *shape):
    # `tf.random_normal` requires integers!
    return randn(dtype, *[convert(s, int) for s in shape])


@dispatch(TFNumeric, Int)
def choice(a, n):
    inds = tf.random_uniform([n], minval=0, maxval=a.shape[0], dtype=tf.int64)
    choices = tf.gather(a, inds)
    return choices[0] if n == 1 else choices
