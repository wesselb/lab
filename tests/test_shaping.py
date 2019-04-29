# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import lab as B
from . import check_function, Tensor, Value, Matrix, List, Tuple
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, lam, allclose, approx


def test_sizing():
    for f in [B.shape, B.shape_int, B.rank, B.length]:
        yield check_function, f, (Tensor(),), {}
        yield check_function, f, (Tensor(3, ),), {}
        yield check_function, f, (Tensor(3, 4),), {}
        yield check_function, f, (Tensor(3, 4, 5),), {}


def test_is_scalar():
    for x in Tensor().forms():
        yield ok, B.is_scalar(x)

    for x in Tensor(3).forms():
        yield ok, not B.is_scalar(x)


def test_expand_dims():
    yield check_function, B.expand_dims, \
          (Tensor(3, 4, 5),), {'axis': Value(0, 1)}


def test_diag():
    yield check_function, B.diag, (Tensor(3),), {}
    yield check_function, B.diag, (Tensor(3, 3),), {}
    yield raises, ValueError, lambda: B.diag(Tensor().tf())


def test_flatten():
    yield check_function, B.flatten, (Tensor(3),), {}
    yield check_function, B.flatten, (Tensor(3, 4),), {}


def test_vec_to_tril_and_back():
    yield check_function, B.vec_to_tril, (Tensor(6),), {}
    yield check_function, B.tril_to_vec, (Matrix(3),), {}

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
          (List(Matrix(3), Matrix(3), Matrix(3)),), {'axis': Value(0, 1)}
    yield check_function, B.stack, \
          (Tuple(Matrix(3), Matrix(3), Matrix(3)),), {'axis': Value(0, 1)}


def test_unstack():
    yield check_function, B.unstack, \
          (Tensor(3, 4, 5),), {'axis': Value(0, 1, 2)}


def test_reshape():
    yield check_function, B.reshape, \
          (Tensor(3, 4, 5),), {'shape': Value((3, 20), (12, 5))}


def test_concat():
    yield check_function, B.concat, \
          (List(Matrix(3), Matrix(3), Matrix(3)),), {'axis': Value(0, 1)}
    yield check_function, B.concat, \
          (Tuple(Matrix(3), Matrix(3), Matrix(3)),), {'axis': Value(0, 1)}


def test_concat2d():
    yield check_function, B.concat2d, \
          (List(List(Matrix(3), Matrix(3)), List(Matrix(3), Matrix(3))),), {}
    yield check_function, B.concat2d, \
          (Tuple(Tuple(Matrix(3), Matrix(3)), Tuple(Matrix(3), Matrix(3))),), {}


def test_take():
    yield check_function, B.take, \
          (Matrix(3, 4), Value([0, 1])), {'axis': Value(0, 1)}
    yield check_function, B.take, \
          (Matrix(3, 4), Value([2, 1])), {'axis': Value(0, 1)}
