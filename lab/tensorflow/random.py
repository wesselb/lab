import logging

import tensorflow as tf

from . import dispatch, B, Numeric
from ..random import _randcat_last_first
from ..types import TFDType, TFNumeric, Int, TFRandomState
from ..util import compress_batch, broadcast_shapes

__all__ = []

log = logging.getLogger(__name__)


@dispatch
def create_random_state(_: TFDType, seed: Int = 0):
    return tf.random.Generator.from_seed(seed)


@dispatch
def global_random_state(_: TFDType):
    return tf.random.get_global_generator()


@dispatch
def set_global_random_state(state: TFRandomState):
    tf.random.set_global_generator(state)


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
def randcat(state: TFRandomState, p: TFNumeric, n: Int):
    # `p` must be at least rank two.
    if B.rank(p) == 1:
        p = B.expand_dims(p, axis=0)
        extra_dim = True
    else:
        extra_dim = False

    p, uncompress = compress_batch(p, 1)
    inds = tf.random.stateless_categorical(
        tf.math.log(p),
        n,
        state.make_seeds()[:, 0],
    )
    inds = uncompress(inds)

    # Possibly remove the extra dimension. Do this before moving the last dimension
    # first!
    if extra_dim:
        inds = inds[0, :]

    inds = _randcat_last_first(inds)

    return state, inds


@dispatch
def randcat(p: TFNumeric, *shape: Int):
    return randcat(global_random_state(p), p, *shape)[1]


@dispatch
def randint(
    state: TFRandomState,
    dtype: TFDType,
    *shape: Int,
    lower: Int = 0,
    upper: Int,
):
    dtype = B.dtype_int(dtype)
    return state, state.uniform(shape, lower, upper, dtype=dtype)


@dispatch
def randint(dtype: TFDType, *shape: Int, lower: Int = 0, upper: Int):
    state = global_random_state(dtype)
    return randint(state, dtype, *shape, lower=lower, upper=upper)[1]


@dispatch
def randperm(state: TFRandomState, dtype: TFDType, n: Int):
    dtype = B.dtype_int(dtype)
    # TF does not have a function to generate a random permutation. One way to do it
    # manually is to generate a range of length `n` and then shuffle it, but TF also
    # does not have a stateless shuffle. Hence, to get a stateless random permutation,
    # we generate random numbers and sort them...
    # TODO: Do this in a better way.
    perm = tf.argsort(state.uniform((n,), dtype=tf.float32))
    return state, B.cast(dtype, perm)


@dispatch
def randperm(dtype: TFDType, n: Int):
    return randperm(global_random_state(dtype), dtype, n)[1]


@dispatch
def randgamma(
    state: TFRandomState,
    dtype: TFDType,
    *shape: Int,
    alpha: Numeric,
    scale: Numeric,
):
    sample = tf.random.stateless_gamma(
        shape + broadcast_shapes(B.shape(alpha), B.shape(scale)),
        alpha=alpha,
        seed=state.make_seeds()[:, 0],
        dtype=dtype,
    )
    return state, sample * B.to_active_device(B.cast(dtype, scale))


@dispatch
def randgamma(dtype: TFDType, *shape: Int, alpha: Numeric, scale: Numeric):
    state = global_random_state(dtype)
    return randgamma(state, dtype, *shape, alpha=alpha, scale=scale)[1]
