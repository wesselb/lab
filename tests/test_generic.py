# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import scipy.special
import numpy as np
import tensorflow as tf
import torch

import lab as B
from . import check_function, Tensor, Value, default_dtype, PositiveTensor, \
    BoolTensor, NaNTensor
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, allclose, approx, eeq


def test_constants():
    yield eq, B.pi, np.pi
    yield eq, B.log_2_pi, np.log(2 * np.pi)
    yield eeq, B.nan, np.nan


def test_isnan():
    yield check_function, B.isnan, (NaNTensor(),), {}
    yield check_function, B.isnan, (NaNTensor(2),), {}
    yield check_function, B.isnan, (NaNTensor(2, 3),), {}


def test_zeros_ones_eye():
    for f in [B.zeros, B.ones, B.eye]:
        # Check consistency.
        yield check_function, f, \
              (Value((2, 3)), Value(np.float32, tf.float32, torch.float32)), {}

        for t1, t2 in [(np.float32, np.int64),
                       (tf.float32, tf.int64),
                       (torch.float32, torch.int64)]:
            ref = B.randn((4, 5), t1)

            # Check shape of shape calls.
            yield eq, B.shape_int(f((2, 3))), (2, 3)
            yield eq, B.shape_int(f((2, 3), t2)), (2, 3)
            yield eq, B.shape_int(f(ref, t2)), (4, 5)
            yield eq, B.shape_int(f((2, 3), ref)), (2, 3)
            yield eq, B.shape_int(f(ref)), (4, 5)

            # Check shape of integer calls.
            if f is B.eye:
                yield eq, B.shape_int(f(3)), (3, 3)
                yield eq, B.shape_int(f(3, t2)), (3, 3)
                yield eq, B.shape_int(f(3, ref)), (3, 3)
            else:
                yield eq, B.shape_int(f(3)), (3,)
                yield eq, B.shape_int(f(3, t2)), (3,)
                yield eq, B.shape_int(f(3, ref)), (3,)

            # Check data type of shape calls.
            yield eeq, B.dtype(f((2, 3))), default_dtype
            yield eeq, B.dtype(f((2, 3), t2)), t2
            yield eeq, B.dtype(f(ref, t2)), t2
            yield eeq, B.dtype(f((2, 3), ref)), t1
            yield eeq, B.dtype(f(ref)), t1

            # Check data type of integer calls.
            yield eeq, B.dtype(f(3)), default_dtype
            yield eeq, B.dtype(f(3, t2)), t2
            yield eeq, B.dtype(f(3, ref)), t1

    # Check exceptions.
    for t in [np.float32, tf.float32, torch.float32]:
        yield raises, ValueError, lambda: B.eye((3,), t)
        yield raises, ValueError, lambda: B.eye((3, 4, 5), t)


def test_cast():
    yield eeq, B.dtype(B.cast(1, np.float64)), np.float64
    yield eeq, B.dtype(B.cast(np.array(1), np.float64)), np.float64
    yield eeq, B.dtype(B.cast(tf.constant(1), tf.float64)), tf.float64
    yield eeq, B.dtype(B.cast(torch.tensor(1), torch.float64)), torch.float64


def test_unary():
    # Test functions with signed arguments.
    for f in [B.abs, B.exp, B.sin, B.cos, B.tan, B.tanh, B.sigmoid, B.relu]:
        yield check_function, f, (Tensor(),), {}
        yield check_function, f, (Tensor(2),), {}
        yield check_function, f, (Tensor(2, 3),), {}

    # Test functions with positive arguments.
    for f in [B.log]:
        yield check_function, f, (PositiveTensor(),), {}
        yield check_function, f, (PositiveTensor(2),), {}
        yield check_function, f, (PositiveTensor(2, 3),), {}


def test_binary():
    for f in [B.add, B.subtract, B.multiply, B.divide, B.power,
              B.minimum, B.maximum, B.leaky_relu]:
        yield check_function, f, (Tensor(), Tensor()), {}
        yield check_function, f, (Tensor(2), Tensor(2)), {}
        yield check_function, f, (Tensor(2, 3), Tensor(2, 3)), {}


def test_reductions():
    for f in [B.min, B.max, B.sum, B.mean, B.std, B.logsumexp]:
        yield check_function, f, (Tensor(),), {}
        yield check_function, f, (Tensor(2),), {}
        yield check_function, f, (Tensor(2),), {'axis': Value(0)}
        yield check_function, f, (Tensor(2, 3),), {}
        yield check_function, f, (Tensor(2, 3),), {'axis': Value(0, 1)}

    # Check correctness of `logsumexp`.
    mat = PositiveTensor(3, 4).np()
    yield allclose, \
          B.logsumexp(mat, axis=1), scipy.special.logsumexp(mat, axis=1)


def test_logical_reductions():
    for f in [B.all, B.any]:
        yield check_function, f, (BoolTensor(),), {}
        yield check_function, f, (BoolTensor(2),), {}
        yield check_function, f, (BoolTensor(2),), {'axis': Value(0)}
        yield check_function, f, (BoolTensor(2, 3),), {}
        yield check_function, f, (BoolTensor(2, 3),), {'axis': Value(0, 1)}


def test_logical_comparisons():
    for f in [B.lt, B.le, B.gt, B.ge]:
        yield check_function, f, (Tensor(), Tensor()), {}
        yield check_function, f, (Tensor(2), Tensor(2)), {}
        yield check_function, f, (Tensor(2, 3), Tensor(2, 3)), {}
