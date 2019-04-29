# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import lab as B
from . import check_function, Tensor, Value
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


# ----------


def test_reshape():
    yield check_function, B.reshape, \
          (Tensor(3, 4, 5),), {'shape': Value((3, 20), (12, 5))}
