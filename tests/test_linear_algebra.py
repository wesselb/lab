# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import lab as B
from . import check_function, Matrix, Bool, Value, PSD, Tensor
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, lam, allclose, approx


def test_transpose():
    yield check_function, B.transpose, (Matrix(),), {}
    yield check_function, B.T, (Matrix(),), {}


def test_matmul():
    yield check_function, B.matmul, \
          (Matrix(), Matrix()), {'tr_a': Bool(), 'tr_b': Bool()}
    yield check_function, B.mm, \
          (Matrix(), Matrix()), {'tr_a': Bool(), 'tr_b': Bool()}
    yield check_function, B.dot, \
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

#
# def test_cholesky_solve():
#     chol = B.cholesky(PSD().np())
#     yield check_function, B.cholesky_solve, (Matrix(mat=chol), Matrix())
#
#
# def test_trisolve():
#     chol = B.cholesky(PSD().np())
#     yield check_function, B.trisolve, \
#           (Matrix(mat=chol), Matrix()), {'tr_a': Bool(), 'lower_a': True}
#     yield check_function, B.trisolve, \
#           (Matrix(mat=chol.T), Matrix()), {'tr_a': Bool(), 'lower_a': False}
