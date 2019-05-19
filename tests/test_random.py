# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import warnings
import numpy  as np
import tensorflow as tf
import torch

import lab as B
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, allclose, approx, \
    assert_isinstance, deq, to_np


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
        yield deq, B.dtype(f()), B.default_dtype
        yield eq, B.shape(f()), ()
        yield deq, B.dtype(f(2)), B.default_dtype
        yield allclose, B.shape(f(2)), (2,)
        yield deq, B.dtype(f(2, 3)), B.default_dtype
        yield eq, B.shape(f(2, 3)), (2, 3)

        # Test with specifying data type.
        for t in [np.float32, tf.float32, torch.float32]:
            # Test direct specification.
            yield deq, B.dtype(f(t)), t
            yield eq, B.shape(f(t)), ()
            yield deq, B.dtype(f(t, 2)), t
            yield eq, B.shape(f(t, 2)), (2,)
            yield deq, B.dtype(f(t, 2, 3)), t
            yield eq, B.shape(f(t, 2, 3)), (2, 3)

            # Test reference specification.
            yield deq, B.dtype(f(f(t))), t
            yield eq, B.shape(f(f())), ()
            yield deq, B.dtype(f(f(t, 2))), t
            yield eq, B.shape(f(f(t, 2))), (2,)
            yield deq, B.dtype(f(f(t, 2, 3))), t
            yield eq, B.shape(f(f(t, 2, 3))), (2, 3)


def test_conversion_warnings():
    for f in [B.rand, B.randn]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            # Trigger the warning!
            f(int, 5)

            yield eq, len(w), 1
