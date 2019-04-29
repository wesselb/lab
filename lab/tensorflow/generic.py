# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from . import dispatch
from ..types import TFNumeric

__all__ = []


@dispatch(TFNumeric)
def abs(a):
    return tf.abs(a)
