# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from . import dispatch
from ..types import TFShape, TFDType

__all__ = []


@dispatch(TFShape, TFDType)
def rand(shape, dtype):
    return tf.random_uniform(shape, dtype=dtype)


@dispatch(TFShape, TFDType)
def randn(shape, dtype):
    return tf.random_normal(shape, dtype=dtype)
