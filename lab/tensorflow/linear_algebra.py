# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from . import dispatch, Numeric

__all__ = []


@dispatch(Numeric, Numeric)
def matmul(a, b, tr_a=False, tr_b=False):
    return tf.matmul(a, b, transpose_a=tr_a, transpose_b=tr_b)


@dispatch(Numeric)
def transpose(a):
    return tf.transpose(a)
