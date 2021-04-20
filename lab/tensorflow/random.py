import logging

import tensorflow as tf

from . import dispatch
from ..types import TFDType, TFNumeric, Int

__all__ = []

log = logging.getLogger(__name__)


@dispatch
def rand(dtype: TFDType, *shape: Int):
    return tf.random.uniform(shape, dtype=dtype)


@dispatch
def randn(dtype: TFDType, *shape: Int):
    return tf.random.normal(shape, dtype=dtype)


@dispatch
def choice(a: TFNumeric, n: Int):
    inds = tf.random.uniform([n], minval=0, maxval=a.shape[0], dtype=tf.int64)
    choices = tf.gather(a, inds)
    return choices[0] if n == 1 else choices
