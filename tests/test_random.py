# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import warnings
import numpy  as np
import tensorflow as tf
import torch

import lab as B
from . import Tensor, to_np
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


def test_choice():
    for x in Tensor(2).forms() + Tensor(2, 3).forms() + Tensor(2, 3, 4).forms():
        # Check shape.
        yield eq, B.shape(B.choice(x)), B.shape(x)[1:]
        yield eq, B.shape(B.choice(x, 1)), B.shape(x)[1:]
        yield eq, B.shape(B.choice(x, 5))[0], 5
        yield eq, B.shape(B.choice(x, 5))[1:], B.shape(x)[1:]

        # Check correctness.
        dtype = B.dtype(x)
        choices = set(to_np(B.choice(B.range(dtype, 5), 1000)))
        yield eq, choices, set(to_np(B.range(dtype, 5)))
