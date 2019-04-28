# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from . import dispatch, TF

__all__ = []


@dispatch(TF)
def abs(a):
    return tf.abs(a)
