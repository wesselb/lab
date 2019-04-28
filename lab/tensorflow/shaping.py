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
