# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import torch

import lab as B
from . import check_function, Tensor, Value, default_dtype
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, allclose, approx, eeq


def test_zeros_ones():
    for f in [B.zeros, B.ones]:
        # Check consistency.
        yield check_function, f, \
              (Value((2, 3)), Value(np.float32, tf.float32, torch.float32)), {}

        for t1, t2 in [(np.float32, np.int64),
                       (tf.float32, tf.int64),
                       (torch.float32, torch.int64)]:
            ref = B.randn((4, 5), t1)

            # Check shape of calls.
            yield eq, B.shape_int(f((2, 3))), (2, 3)
            yield eq, B.shape_int(f((2, 3), t2)), (2, 3)
            yield eq, B.shape_int(f(ref, t2)), (4, 5)
            yield eq, B.shape_int(f((2, 3), ref)), (2, 3)
            yield eq, B.shape_int(f(ref)), (4, 5)

            # Check dtype of calls.
            yield eeq, B.dtype(f((2, 3))), default_dtype
            yield eeq, B.dtype(f((2, 3), t2)), t2
            yield eeq, B.dtype(f(ref, t2)), t2
            yield eeq, B.dtype(f((2, 3), ref)), t1
            yield eeq, B.dtype(f(ref)), t1


def test_cast():
    yield eeq, B.dtype(B.cast(1, np.float64)), np.float64
    yield eeq, B.dtype(B.cast(np.array(1), np.float64)), np.float64
    yield eeq, B.dtype(B.cast(tf.constant(1), tf.float64)), tf.float64
    yield eeq, B.dtype(B.cast(torch.tensor(1), torch.float64)), torch.float64


def test_abs():
    yield check_function, B.abs, (Tensor(2, 3, 4),), {}
