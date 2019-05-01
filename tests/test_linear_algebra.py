# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

import lab as B
from . import check_function, Matrix, Bool, Value, PSD, Tensor
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, allclose, approx


def test_constants():
    yield eq, B.epsilon, 1e-12


def test_transpose():
    for f in [B.transpose, B.T]:
        yield check_function, f, (Matrix(),), {}
        yield check_function, f, (Matrix(),), {}


def test_matmul():
    for f in [B.matmul, B.mm, B.dot]:
        yield check_function, f, \
              (Matrix(), Matrix()), {'tr_a': Bool(), 'tr_b': Bool()}


def test_trace():
    yield check_function, B.trace, \
          (Matrix(),), {'axis1': Value(0), 'axis2': Value(1)}
    yield check_function, B.trace, \
          (Matrix(),), {'axis1': Value(1), 'axis2': Value(0)}


def test_kron():
    yield check_function, B.kron, (Tensor(2, 3), Tensor(4, 5)), {}
    yield raises, ValueError, \
          lambda: B.kron(Tensor(2).tf(), Tensor(4, 5).tf())
    yield raises, ValueError, \
          lambda: B.kron(Tensor(2).torch(), Tensor(4, 5).torch())


def test_svd():
    # Take absolute value because the sign of the result is undetermined.
    def svd(a, compute_uv=True):
        if compute_uv:
            u, s, v = B.svd(a, compute_uv=True)
            return B.abs(u), s, B.abs(v)
        else:
            return B.svd(a, compute_uv=False)

    yield check_function, svd, (Matrix(),), {'compute_uv': Bool()}


def test_cholesky():
    yield check_function, B.cholesky, (PSD(),), {}


def test_cholesky_solve():
    chol = B.cholesky(PSD().np())
    yield check_function, B.cholesky_solve, (Matrix(mat=chol), Matrix()), {}


def test_trisolve():
    chol = B.cholesky(PSD().np())
    yield check_function, B.trisolve, \
          (Matrix(mat=chol), Matrix()), {'lower_a': Value(True)}
    yield check_function, B.trisolve, \
          (Matrix(mat=chol.T), Matrix()), {'lower_a': Value(False)}


def test_outer():
    yield raises, ValueError, lambda: B.outer(B.eye(5), B.ones(5))
    yield raises, ValueError, lambda: B.outer(B.ones(5), B.eye(5))
    a, b = B.randn(5), B.randn(5)
    yield allclose, B.outer(a), np.outer(a, a)
    yield allclose, B.outer(a, b), np.outer(a, b)
