# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from . import dispatch
from ..types import TFDimension, TFDType

__all__ = []


@dispatch(TFDType, [TFDimension])
def rand(dtype, *shape):
    return tf.random_uniform(shape, dtype=dtype)


@dispatch(TFDType, [TFDimension])
def randn(dtype, *shape):
    return tf.random_normal(shape, dtype=dtype)
