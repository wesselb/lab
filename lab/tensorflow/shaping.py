# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from . import dispatch
from ..types import TFNumeric, TFListOrTuple

__all__ = []


@dispatch(TFNumeric)
def shape(a):
    s = tf.shape(a)
    return tuple(s[i] for i in range(rank(a)))


@dispatch(TFNumeric)
def shape_int(a):
    return tuple(x.value for x in a.get_shape())


@dispatch(TFNumeric)
def rank(a):
    return len(shape_int(a))


@dispatch(TFNumeric)
def length(a):
    return tf.size(a)


@dispatch(TFNumeric)
def expand_dims(a, axis=0):
    return tf.expand_dims(a, axis=axis)


@dispatch(TFNumeric)
def diag(a):
    if rank(a) == 1:
        return tf.diag(a)
    elif rank(a) == 2:
        return tf.diag_part(a)
    else:
        raise ValueError('Argument must have rank 1 or 2.')


@dispatch(TFNumeric)
def vec_to_tril(a):
    if rank(a) != 1:
        raise ValueError('Input must be rank 1.')

    # Figure out shape of output.
    n = shape_int(a)[0]
    m = int(((1 + 8 * n) ** .5 - 1) / 2)

    # Construct output and return.
    return tf.scatter_nd(indices=list(zip(*np.tril_indices(m))),
                         shape=[m, m],
                         updates=a)


@dispatch(TFNumeric)
def tril_to_vec(a):
    if rank(a) != 2:
        raise ValueError('Input must be rank 2.')
    n, m = shape_int(a)
    if n != m:
        raise ValueError('Input must be square.')
    return tf.gather_nd(a, list(zip(*np.tril_indices(n))))


@dispatch(TFListOrTuple)
def stack(a, axis=0):
    return tf.stack(a, axis=axis)


# -------

@dispatch(TFNumeric)
def reshape(a, shape=(-1,)):
    return tf.reshape(a, shape=shape)
