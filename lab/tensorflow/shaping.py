# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from . import dispatch, TF

__all__ = []


@dispatch(TF)
def shape(a):
    s = tf.shape(a)
    return tuple(s[i] for i in range(rank(a)))


@dispatch(TF)
def shape_int(a):
    return tuple(x.value for x in a.get_shape())


@dispatch(TF)
def rank(a):
    return len(shape_int(a))


@dispatch(TF)
def length(a):
    return tf.size(a)


@dispatch(TF)
def expand_dims(a, axis=0):
    return tf.expand_dims(a, axis=axis)


@dispatch(TF)
def diag(a):
    if rank(a) == 1:
        return tf.diag(a)
    elif rank(a) == 2:
        return tf.diag_part(a)
    else:
        raise ValueError('Argument must have rank 1 or 2.')


# -------

@dispatch(TF)
def reshape(a, shape=(-1,)):
    return tf.reshape(a, shape=shape)
