# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from . import dispatch, Numeric

__all__ = []


@dispatch(Numeric)
def shape(a):
    s = tf.shape(a)
    return tuple(s[i] for i in range(rank(a)))


@dispatch(Numeric)
def shape_int(a):
    return tuple(x.value for x in a.get_shape())


@dispatch(Numeric)
def rank(a):
    return len(shape_int(a))


@dispatch(Numeric)
def length(a):
    return tf.size(a)
