# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from . import dispatch
from ..shaping import _vec_to_tril_shape
from ..types import TFNumeric, TFDimension

__all__ = []


@dispatch(TFNumeric)
def shape(a):
    return tuple(a.shape)


@dispatch(TFNumeric)
def shape_int(a):
    return tuple(x.value for x in shape(a))


@dispatch(TFNumeric)
def rank(a):
    return len(a.shape)


@dispatch(TFNumeric)
def length(a):
    return tf.size(a)


@dispatch(TFNumeric)
def expand_dims(a, axis=0):
    return tf.expand_dims(a, axis=axis)


@dispatch(TFNumeric)
def squeeze(a):
    return tf.squeeze(a)


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
    m = _vec_to_tril_shape(a)
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


@dispatch([TFNumeric])
def stack(*elements, **kw_args):
    return tf.stack(elements, axis=kw_args.get('axis', 0))


@dispatch(TFNumeric)
def unstack(a, axis=0):
    return tf.unstack(a, axis=axis)


@dispatch(TFNumeric, [TFDimension])
def reshape(a, *shape):
    return tf.reshape(a, shape=shape)


@dispatch([TFNumeric])
def concat(*elements, **kw_args):
    return tf.concat(elements, axis=kw_args.get('axis', 0))


@dispatch(TFNumeric, object)
def take(a, indices, axis=0):
    # Optimise the case where `axis` equals `0`.
    if axis == 0:
        return tf.gather(a, indices)

    # Create a permutation to switch `axis` and `0`.
    perm = [i for i in range(rank(a))]
    perm[axis], perm[0] = 0, axis

    # Perform gathering.
    return tf.transpose(tf.gather(tf.transpose(a, perm), indices), perm)
