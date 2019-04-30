# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy  as np
import tensorflow as tf
import torch
import lab as B
from . import check_function, Tensor
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, allclose, approx, eeq


def test_cast():
    yield eeq, B.dtype(B.cast(1, np.float64)), np.float64
    yield eeq, B.dtype(B.cast(np.array(1), np.float64)), np.float64
    yield eeq, B.dtype(B.cast(tf.constant(1), tf.float64)), tf.float64
    yield eeq, B.dtype(B.cast(torch.tensor(1), torch.float64)), torch.float64


def test_abs():
    yield check_function, B.abs, (Tensor(2, 3, 4),), {}
