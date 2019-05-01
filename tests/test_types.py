# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from autograd import grad
import tensorflow as tf
import torch

from plum.promotion import _promotion_rule, convert

import lab as B
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, allclose, approx, \
    assert_isinstance, eeq


def test_numeric():
    # Test convenient types.
    yield assert_isinstance, 1, B.Int
    yield assert_isinstance, np.int32(1), B.Int
    yield assert_isinstance, np.uint64(1), B.Int
    yield assert_isinstance, 1.0, B.Float
    yield assert_isinstance, np.float32(1), B.Float
    yield assert_isinstance, True, B.Bool
    yield assert_isinstance, np.bool_(True), B.Bool
    yield assert_isinstance, np.uint(1), B.Number
    yield assert_isinstance, np.float64(1), B.Number

    # Test NumPy.
    yield assert_isinstance, 1, B.NPNumeric
    yield assert_isinstance, np.bool_(1), B.NPNumeric
    yield assert_isinstance, np.float32(1), B.NPNumeric
    yield assert_isinstance, np.array(1), B.NPNumeric

    # Test TensorFlow.
    yield assert_isinstance, tf.constant(1), B.TFNumeric
    yield assert_isinstance, tf.Variable(1), B.TFNumeric

    # Test Torch.
    yield assert_isinstance, torch.tensor(1), B.TorchNumeric

    # Test general numeric type.
    yield assert_isinstance, 1, B.Numeric
    yield assert_isinstance, np.bool_(1), B.Numeric
    yield assert_isinstance, np.float64(1), B.Numeric
    yield assert_isinstance, np.array(1), B.Numeric
    yield assert_isinstance, tf.constant(1), B.Numeric
    yield assert_isinstance, torch.tensor(1), B.Numeric

    # Test promotion.
    yield eq, _promotion_rule(np.array(1), tf.constant(1)), B.TFNumeric
    yield eq, _promotion_rule(np.array(1), tf.Variable(1)), B.TFNumeric
    yield eq, _promotion_rule(tf.constant(1), tf.Variable(1)), B.TFNumeric
    yield eq, _promotion_rule(np.array(1), torch.tensor(1)), B.TorchNumeric
    yield raises, TypeError, \
          lambda: _promotion_rule(B.TFNumeric, B.TorchNumeric)

    # Test conversion.
    yield assert_isinstance, convert(np.array(1), B.TFNumeric), B.TFNumeric
    yield assert_isinstance, convert(np.array(1), B.TorchNumeric), \
          B.TorchNumeric


def test_autograd_tracing():
    found_objs = []

    def f(x):
        found_objs.append(x)
        return B.sum(x)

    # Test that function runs.
    yield f, np.ones(5)
    yield grad(f), np.ones(5)

    # Test that objects are of the right type.
    for obj in found_objs:
        yield assert_isinstance, obj, B.NPNumeric


def test_tuple():
    for t in [B.NPTuple, B.NPListOrTuple, B.ListOrTuple]:
        yield assert_isinstance, (1, np.bool_(True)), t
        yield assert_isinstance, (np.float32(1), np.array(1)), t

    for t in [B.TFTuple, B.TFListOrTuple, B.ListOrTuple]:
        yield assert_isinstance, (tf.constant(1), tf.constant(2)), t
        yield assert_isinstance, (tf.constant(1), tf.Variable(2)), t
        yield assert_isinstance, (tf.Variable(1), tf.constant(2)), t
        yield assert_isinstance, (tf.Variable(1), tf.Variable(2)), t

    for t in [B.TorchTuple, B.TorchListOrTuple, B.ListOrTuple]:
        yield assert_isinstance, (torch.tensor(1), torch.tensor(2)), t


def test_list():
    for t in [B.NPList, B.NPListOrTuple, B.ListOrTuple]:
        yield assert_isinstance, [1, np.bool_(True)], t
        yield assert_isinstance, [np.float32(1), np.array(1)], t

    for t in [B.TFList, B.TFListOrTuple, B.ListOrTuple]:
        yield assert_isinstance, [tf.constant(1), tf.constant(2)], t
        yield assert_isinstance, [tf.constant(1), tf.Variable(2)], t
        yield assert_isinstance, [tf.Variable(1), tf.constant(2)], t
        yield assert_isinstance, [tf.Variable(1), tf.Variable(2)], t

    for t in [B.TorchList, B.TorchListOrTuple, B.ListOrTuple]:
        yield assert_isinstance, [torch.tensor(1), torch.tensor(2)], t


def test_shape():
    for t in [B.NPShape, B.TFShape, B.TorchShape]:
        yield assert_isinstance, [], t
        yield assert_isinstance, [1, np.uint(8)], t
        yield assert_isinstance, (), t
        yield assert_isinstance, (np.int32(1), 2), t

    # Test TensorFlow-specific and PyTorch-specific shapes.
    for a, t in [(tf.random_normal([2, 2]), B.TFShape),
                 (torch.randn(2, 2), B.TorchShape)]:
        yield assert_isinstance, a.shape, t
        yield assert_isinstance, (a.shape[0], a.shape[1]), t
        yield assert_isinstance, [a.shape[0], a.shape[1]], t

    # Test NumPy-specific shapes.
    yield assert_isinstance, (np.int32(1), np.int32(1)), B.NPShape
    yield assert_isinstance, (np.int64(1), np.int64(1)), B.NPShape


def test_data_type():
    yield assert_isinstance, np.float32, B.NPDType
    yield assert_isinstance, np.float32, B.DType
    yield assert_isinstance, tf.float32, B.TFDType
    yield assert_isinstance, tf.float32, B.DType
    yield assert_isinstance, torch.float32, B.TorchDType
    yield assert_isinstance, torch.float32, B.DType

    # Test conversion between data types.
    yield eeq, convert(np.float32, B.TFDType), tf.float32
    yield eeq, convert(np.float32, B.TorchDType), torch.float32
    yield eeq, convert(tf.float32, B.NPDType), np.float32
    yield eeq, convert(tf.float32, B.TorchDType), torch.float32
    yield eeq, convert(torch.float32, B.NPDType), np.float32
    yield eeq, convert(torch.float32, B.TFDType), tf.float32

    # Test conversion of `np.dtype`.
    yield eeq, convert(np.dtype('float32'), B.DType), np.float32


def test_issubdtype():
    yield call, B.issubdtype, (np.float32, np.floating)
    yield call, B.issubdtype, (tf.float32, np.floating)
    yield call, B.issubdtype, (torch.float32, np.floating)
    yield call, B.issubdtype, (np.float32, np.integer), {}, False
    yield call, B.issubdtype, (tf.float32, np.integer), {}, False
    yield call, B.issubdtype, (torch.float32, np.integer), {}, False


def test_dtype():
    yield call, B.dtype, (1,), {}, int
    yield call, B.dtype, (1.0,), {}, float
    yield call, B.dtype, (np.array(1, dtype=np.int32),), {}, np.int32
    yield call, B.dtype, (np.array(1.0, dtype=np.float32),), {}, np.float32
    yield call, B.dtype, (tf.constant(1, dtype=tf.int32),), {}, tf.int32
    yield call, B.dtype, (tf.constant(1.0, dtype=tf.float32),), {}, tf.float32
    yield call, B.dtype, (torch.tensor(1, dtype=torch.int32),), {}, \
          torch.int32
    yield call, B.dtype, (torch.tensor(1.0, dtype=torch.float32),), {}, \
          torch.float32


def test_framework():
    for t in [B.NP, B.Framework]:
        yield assert_isinstance, np.array(1), t
        yield assert_isinstance, (np.array(1)), t
        yield assert_isinstance, [np.array(1)], t
        yield assert_isinstance, np.ones((5, 5)).shape, t
        yield assert_isinstance, np.float32, t

    for t in [B.TF, B.Framework]:
        yield assert_isinstance, tf.constant(1), t
        yield assert_isinstance, (tf.constant(1)), t
        yield assert_isinstance, [tf.constant(1)], t
        yield assert_isinstance, tf.ones([5, 5]).shape, t
        yield assert_isinstance, tf.float32, t

    for t in [B.Torch, B.Framework]:
        yield assert_isinstance, torch.tensor(1), t
        yield assert_isinstance, (torch.tensor(1)), t
        yield assert_isinstance, [torch.tensor(1)], t
        yield assert_isinstance, torch.ones(5, 5).shape, t
        yield assert_isinstance, torch.float32, t
