# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from . import dispatch
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
