# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy  as np
import tensorflow as tf
import torch

import lab as B
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, allclose, approx, \
    assert_isinstance, eeq, to_np


def test_set_seed():
    for t in [np.float32, tf.float32, torch.float32]:
        # Careful with TensorFlow's graph!
        tf.reset_default_graph()
        B.set_random_seed(0)
        x = to_np(B.rand(t))
        tf.reset_default_graph()
        B.set_random_seed(0)
        y = to_np(B.rand(t))
        yield eq, x, y


def test_random_generators():
    for f in [B.rand, B.randn]:
        # Test without specifying data type.
        yield eeq, B.dtype(f()), B.default_dtype
        yield eeq, B.dtype(f(())), B.default_dtype
        yield eeq, B.dtype(f((2,))), B.default_dtype
        yield eeq, B.dtype(f((2, 2))), B.default_dtype

        # Test with specifying data type.
        for t in [np.float32, tf.float32, torch.float32]:
            yield eeq, B.dtype(f(t)), t
            yield eeq, B.dtype(f((), t)), t
            yield eeq, B.dtype(f((2,), t)), t
            yield eeq, B.dtype(f((2, 2), t)), t
