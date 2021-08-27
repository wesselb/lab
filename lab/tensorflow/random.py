import logging

import tensorflow as tf
from plum import Union, Val

from . import dispatch
from ..types import TFDType, TFNumeric, Int, TFRandomState

__all__ = []

log = logging.getLogger(__name__)


@dispatch
def create_random_state(_: TFDType, seed: Int = 0):
    return tf.random.Generator.from_seed(seed)


@dispatch
def rand(state: TFRandomState, dtype: TFDType, *shape: Int):
    return state, state.uniform(shape, dtype=dtype)


@dispatch
def rand(dtype: TFDType, *shape: Int):
    return tf.random.uniform(shape, dtype=dtype)


@dispatch
def randn(state: TFRandomState, dtype: TFDType, *shape: Int):
    return state, state.normal(shape, dtype=dtype)


@dispatch
def randn(dtype: TFDType, *shape: Int):
    return tf.random.normal(shape, dtype=dtype)


@dispatch
def choice(state: TFRandomState, a: TFNumeric, n: Int):
    inds = state.uniform([n], minval=0, maxval=a.shape[0], dtype=tf.int64)
    choices = tf.gather(a, inds)
    return state, choices[0] if n == 1 else choices


@dispatch
def choice(state: Val[tf.random], a: TFNumeric, n: Int):
    method = choice.invoke(TFRandomState, TFNumeric, Int)
    return method(type(state).type_parameter, a, n)


@dispatch
def choice(a: TFNumeric, n: Int):
    return choice(Val(tf.random), a, n)[1]
