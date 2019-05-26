# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from . import dispatch, B
from ..shaping import _vec_to_tril_shape
from ..types import TFNumeric, TFDimension

__all__ = []


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
    if B.rank(a) == 1:
        return tf.diag(a)
    elif B.rank(a) == 2:
        return tf.diag_part(a)
    else:
        raise ValueError('Argument must have rank 1 or 2.')


@dispatch(TFNumeric)
def vec_to_tril(a):
    if B.rank(a) != 1:
        raise ValueError('Input must be rank 1.')
    m = _vec_to_tril_shape(a)
    return tf.scatter_nd(indices=list(zip(*np.tril_indices(m))),
                         shape=[m, m],
                         updates=a)


@dispatch(TFNumeric)
def tril_to_vec(a):
    if B.rank(a) != 2:
        raise ValueError('Input must be rank 2.')
    n, m = B.shape(a)
    if n != m:
        raise ValueError('Input must be square.')
    return tf.gather_nd(a, list(zip(*np.tril_indices(int(n)))))


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


@dispatch(TFNumeric, [TFDimension])
def tile(a, *repeats):
    return tf.tile(a, repeats)


@dispatch(TFNumeric, object)
def take(a, indices_or_mask, axis=0):
    # Put axis `axis` first.
    if axis > 0:
        # Create a permutation to switch `axis` and `0`.
        perm = list(range(B.rank(a)))
        perm[0], perm[axis] = perm[axis], perm[0]
        a = tf.transpose(a, perm)

    # Figure out whether we're given indices or a mask.
    if isinstance(indices_or_mask, B.TF):
        mask = indices_or_mask.dtype == bool
    else:
        mask = len(indices_or_mask) > 0 and B.dtype(indices_or_mask[0]) == bool

    # Take the relevant part.
    if mask:
        a = tf.boolean_mask(a, indices_or_mask)
    else:
        a = tf.gather(a, indices_or_mask)

    # Put axis `axis` back again.
    if axis > 0:
        a = tf.transpose(a, perm)

    return a
