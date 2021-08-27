import logging

import tensorflow as tf
from plum import Union, Val

from . import dispatch
from ..types import TFDType, TFNumeric, Int, TFRandomState

__all__ = []

log = logging.getLogger(__name__)


class _GlobalTFRandomStateProxy(tf.random.Generator):
    def __init__(self):
        pass

    def __getattribute__(self, name):
        return getattr(tf.random, name)


_tf_global_random_state = _GlobalTFRandomStateProxy()


@dispatch
def create_random_state(_: TFDType, seed: Int = 0):
    return tf.random.Generator.from_seed(seed)


@dispatch
def global_random_state(_: TFDType):
    return _tf_global_random_state


@dispatch
def rand(state: TFRandomState, dtype: TFDType, *shape: Int):
    return state, state.uniform(shape, dtype=dtype)


@dispatch
def rand(dtype: TFDType, *shape: Int):
    return rand(global_random_state(dtype), dtype, *shape)[1]


@dispatch
def randn(state: TFRandomState, dtype: TFDType, *shape: Int):
    return state, state.normal(shape, dtype=dtype)


@dispatch
def randn(dtype: TFDType, *shape: Int):
    return randn(global_random_state(dtype), dtype, *shape)[1]


@dispatch
def choice(state: TFRandomState, a: TFNumeric, n: Int):
    inds = state.uniform([n], minval=0, maxval=a.shape[0], dtype=tf.int64)
    choices = tf.gather(a, inds)
    return state, choices[0] if n == 1 else choices


@dispatch
def choice(a: TFNumeric, n: Int):
    return choice(global_random_state(a), a, n)[1]
