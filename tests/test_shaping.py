# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import lab as B
from . import check_function, Tensor, Value, Matrix, List, Tuple
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, allclose, approx


def test_sizing():
    for f in [B.shape, B.rank, B.length, B.size]:
        yield check_function, f, (Tensor(),), {}, False
        yield check_function, f, (Tensor(3, ),), {}, False
        yield check_function, f, (Tensor(3, 4),), {}, False
        yield check_function, f, (Tensor(3, 4, 5),), {}, False


def test_isscalar():
    yield ok, B.isscalar(1.0)
    yield ok, not B.isscalar(np.array([1.0]))


def test_expand_dims():
    yield check_function, B.expand_dims, \
          (Tensor(3, 4, 5),), {'axis': Value(0, 1)}


def test_squeeze():
    yield check_function, B.squeeze, (Tensor(3, 4, 5),)
    yield check_function, B.squeeze, (Tensor(1, 4, 5),)
    yield check_function, B.squeeze, (Tensor(3, 1, 5),)
    yield check_function, B.squeeze, (Tensor(1, 4, 1),)


def test_uprank():
    yield allclose, B.uprank(1.0), np.array([[1.0]])
    yield allclose, B.uprank(np.array([1.0, 2.0])), np.array([[1.0], [2.0]])
    yield allclose, B.uprank(np.array([[1.0, 2.0]])), np.array([[1.0, 2.0]])
    yield raises, ValueError, lambda: B.uprank(np.array([[[1.0]]]))


def test_diag():
    yield check_function, B.diag, (Tensor(3),)
    yield check_function, B.diag, (Tensor(3, 3),)
    yield raises, ValueError, lambda: B.diag(Tensor().tf())


def test_flatten():
    yield check_function, B.flatten, (Tensor(3),)
    yield check_function, B.flatten, (Tensor(3, 4),)


def test_vec_to_tril_and_back():
    yield check_function, B.vec_to_tril, (Tensor(6),)
    yield check_function, B.tril_to_vec, (Matrix(3),)

    # Check correctness.
    mat = Tensor(6).np()
    yield allclose, B.tril_to_vec(B.vec_to_tril(mat)), mat

    # Check exceptions.
    for x in Matrix(3, 4).forms():
        yield raises, ValueError, lambda: B.vec_to_tril(x)
        yield raises, ValueError, lambda: B.tril_to_vec(x)
    for x in Matrix(3, 4, 5).forms():
        yield raises, ValueError, lambda: B.tril_to_vec(x)


def test_stack():
    yield check_function, B.stack, \
          (Matrix(3), Matrix(3), Matrix(3)), {'axis': Value(0, 1)}


def test_unstack():
    yield check_function, B.unstack, \
          (Tensor(3, 4, 5),), {'axis': Value(0, 1, 2)}


def test_reshape():
    yield check_function, B.reshape, (Tensor(3, 4, 5), Value(3), Value(20))
    yield check_function, B.reshape, (Tensor(3, 4, 5), Value(12), Value(5))


def test_concat():
    yield check_function, B.concat, \
          (Matrix(3), Matrix(3), Matrix(3)), {'axis': Value(0, 1)}


def test_concat2d():
    yield check_function, B.concat2d, \
          (List(Matrix(3), Matrix(3)), List(Matrix(3), Matrix(3)))
    yield check_function, B.concat2d, \
          (Tuple(Matrix(3), Matrix(3)), Tuple(Matrix(3), Matrix(3)))


def test_take():
    yield check_function, B.take, \
          (Matrix(3, 4), Value([0, 1])), {'axis': Value(0, 1)}
    yield check_function, B.take, \
          (Matrix(3, 4), Value([2, 1])), {'axis': Value(0, 1)}
