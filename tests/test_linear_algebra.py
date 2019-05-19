# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from itertools import product
import numpy as np

import lab as B
from . import check_function, Matrix, Bool, Value, PSD, Tensor
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, allclose, approx, is_


def test_constants():
    yield eq, B.epsilon, 1e-12


def test_transpose():
    for f in [B.transpose, B.T, B.t]:
        # Check consistency.
        yield check_function, f, (Tensor(),)
        yield check_function, f, (Tensor(2),), {'perm': Value(None, (0,))}
        yield check_function, f, \
              (Tensor(2, 3),), {'perm': Value(None, (0, 1), (1, 0))}
        yield check_function, f, \
              (Tensor(2, 3, 4),), {'perm': Value(None,
                                                 (0, 1, 2),
                                                 (0, 2, 1),
                                                 (1, 0, 2),
                                                 (1, 2, 0),
                                                 (2, 1, 0),
                                                 (2, 0, 1))}

        # Check correctness of zero-dimensional case.
        yield is_, f(1), 1

        # Check correctness of one-dimensional case.
        a = Tensor(2).np()
        yield allclose, f(a, perm=None), a[None, :]
        yield allclose, f(a, perm=(0,)), a

        # Check correctness of three-dimensional case.
        a = Tensor(2, 3, 4).np()
        yield allclose, f(a), np.transpose(a, axes=(0, 2, 1))
        yield allclose, f(a, perm=(1, 2, 0)), np.transpose(a, axes=(1, 2, 0))


def test_matmul():
    for f in [B.matmul, B.mm, B.dot]:
        yield check_function, f, \
              (Matrix(), Matrix()), {'tr_a': Bool(), 'tr_b': Bool()}


def test_trace():
    # Check default call.
    yield check_function, B.trace, \
          (Tensor(2, 3, 4, 5),), {'axis1': Value(0), 'axis2': Value(1)}

    # Check calls with `axis1 < axis2`.
    yield check_function, B.trace, \
          (Tensor(2, 3, 4, 5),), {'axis1': Value(0, 1), 'axis2': Value(2, 3)}

    # Check calls with `axis1 > axis2`.
    yield check_function, B.trace, \
          (Tensor(2, 3, 4, 5),), {'axis1': Value(2, 3), 'axis2': Value(0, 1)}

    # Check that call with `axis1 == axis2` raises an error in NumPy.
    yield raises, ValueError, lambda: B.trace(Matrix().np(), axis1=0, axis2=0)


def test_kron():
    yield check_function, B.kron, (Tensor(2, 3), Tensor(4, 5))
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


def test_solve():
    yield check_function, B.solve, (Matrix(), Matrix())


def test_inv():
    yield check_function, B.inv, (Matrix(),)


def test_det():
    yield check_function, B.det, (Matrix(),)


def test_logdet():
    yield check_function, B.logdet, (PSD(),)


def test_cholesky():
    for f in [B.cholesky, B.chol]:
        yield check_function, f, (PSD(),)


def test_cholesky_solve():
    chol = B.cholesky(PSD().np())
    for f in [B.cholesky_solve, B.cholsolve]:
        yield check_function, f, (Matrix(mat=chol), Matrix())


def test_triangular_solve():
    chol = B.cholesky(PSD().np())
    for f in [B.triangular_solve, B.trisolve]:
        yield check_function, f, \
              (Matrix(mat=chol), Matrix()), {'lower_a': Value(True)}
        yield check_function, f, \
              (Matrix(mat=chol.T), Matrix()), {'lower_a': Value(False)}


def test_toeplitz_solve():
    for f in [B.toeplitz_solve, B.toepsolve]:
        yield check_function, f, (Tensor(3), Matrix(3))
        yield check_function, f, (Tensor(3), Matrix(3))


def test_outer():
    for a, b in list(product(*([[Tensor(5).np(), Tensor(5, 1).np()]] * 2))) + \
                [(Tensor(5, 3).np(), Tensor(5, 3).np())]:
        yield allclose, \
              B.outer(a, b), B.matmul(B.uprank(a), B.uprank(b), tr_b=True)
        yield allclose, B.outer(a), B.outer(a, a)
        yield allclose, B.outer(b), B.outer(b, b)


def test_reg():
    old_epsilon = B.epsilon
    B.epsilon = 10
    a = Matrix(2, 3).np()
    yield allclose, B.reg(a, diag=None, clip=False), a + 10 * np.eye(*a.shape)
    yield allclose, B.reg(a, diag=None, clip=True), a + 10 * np.eye(*a.shape)
    yield allclose, B.reg(a, diag=1, clip=False), a + 1 * np.eye(*a.shape)
    yield allclose, B.reg(a, diag=1, clip=True), a + 10 * np.eye(*a.shape)
    yield allclose, B.reg(a, diag=100, clip=False), a + 100 * np.eye(*a.shape)
    yield allclose, B.reg(a, diag=100, clip=True), a + 100 * np.eye(*a.shape)
    B.epsilon = old_epsilon


def test_pw_2d():
    # In this case, allow for 1e-7 absolute error, because the computation is
    # approximate.
    def approx_allclose(a, b):
        np.testing.assert_allclose(a, b, atol=1e-7)

    a, b = Matrix(5, 2).np(), Matrix(10, 2).np()
    dists2_ab, dists2_aa = np.zeros((5, 10)), np.zeros((5, 5))
    sums2_ab, sums2_aa = np.zeros((5, 10)), np.zeros((5, 5))
    for i in range(5):
        for j in range(10):
            dists2_ab[i, j] = np.sum((a[i] - b[j]) ** 2)
            sums2_ab[i, j] = np.sum((a[i] + b[j]) ** 2)
            if j < 5:
                dists2_aa[i, j] = np.sum((a[i] - a[j]) ** 2)
                sums2_aa[i, j] = np.sum((a[i] + a[j]) ** 2)

    yield approx_allclose, B.pw_dists2(a, b), dists2_ab
    yield approx_allclose, B.pw_dists2(a), dists2_aa
    yield approx_allclose, B.pw_dists(a, b), np.maximum(dists2_ab, 1e-30) ** .5
    yield approx_allclose, B.pw_dists(a), np.maximum(dists2_aa, 1e-30) ** .5
    yield approx_allclose, B.pw_sums2(a, b), sums2_ab
    yield approx_allclose, B.pw_sums2(a), sums2_aa
    yield approx_allclose, B.pw_sums(a, b), np.maximum(sums2_ab, 1e-30) ** .5
    yield approx_allclose, B.pw_sums(a), np.maximum(sums2_aa, 1e-30) ** .5


def test_pw_1d():
    a, b = Matrix(5, 1).np(), Matrix(10, 1).np()

    # Check that we can feed both rank 1 and rank 2 tensors.
    for f, g in product(*([[lambda x: x, lambda x: x[:, 0]]] * 2)):

        yield allclose, B.pw_dists2(f(a), g(b)), np.abs(a - b.T) ** 2
        yield allclose, B.pw_dists2(f(a)), np.abs(a - a.T) ** 2
        yield allclose, B.pw_dists(f(a), g(b)), np.abs(a - b.T)
        yield allclose, B.pw_dists(f(a)), np.abs(a - a.T)
        yield allclose, B.pw_sums2(f(a), g(b)), np.abs(a + b.T) ** 2
        yield allclose, B.pw_sums2(f(a)), np.abs(a + a.T) ** 2
        yield allclose, B.pw_sums(f(a), g(b)), np.abs(a + b.T)
        yield allclose, B.pw_sums(f(a)), np.abs(a + a.T)


def test_ew_2d():
    a, b = Matrix(10, 2).np(), Matrix(10, 2).np()
    dists2_ab, dists2_aa = np.zeros((10, 1)), np.zeros((10, 1))
    sums2_ab, sums2_aa = np.zeros((10, 1)), np.zeros((10, 1))
    for i in range(10):
        dists2_ab[i, 0] = np.sum((a[i] - b[i]) ** 2)
        dists2_aa[i, 0] = np.sum((a[i] - a[i]) ** 2)
        sums2_ab[i, 0] = np.sum((a[i] + b[i]) ** 2)
        sums2_aa[i, 0] = np.sum((a[i] + a[i]) ** 2)

    yield allclose, B.ew_dists2(a, b), dists2_ab
    yield allclose, B.ew_dists2(a), dists2_aa
    yield allclose, B.ew_dists(a, b), np.maximum(dists2_ab, 1e-30) ** .5
    yield allclose, B.ew_dists(a), np.maximum(dists2_aa, 1e-30) ** .5
    yield allclose, B.ew_sums2(a, b), sums2_ab
    yield allclose, B.ew_sums2(a), sums2_aa
    yield allclose, B.ew_sums(a, b), np.maximum(sums2_ab, 1e-30) ** .5
    yield allclose, B.ew_sums(a), np.maximum(sums2_aa, 1e-30) ** .5


def test_ew_1d():
    a, b = Matrix(10, 1).np(), Matrix(10, 1).np()

    # Check that we can feed both rank 1 and rank 2 tensors.
    for f, g in product(*([[lambda x: x, lambda x: x[:, 0]]] * 2)):

        yield allclose, B.ew_dists2(f(a), g(b)), np.abs(a - b) ** 2
        yield allclose, B.ew_dists2(f(a)), np.zeros((10, 1))
        yield allclose, B.ew_dists(f(a), g(b)), np.abs(a - b)
        yield allclose, B.ew_dists(f(a)), np.zeros((10, 1))
        yield allclose, B.ew_sums2(f(a), g(b)), np.abs(a + b) ** 2
        yield allclose, B.ew_sums2(f(a)), np.abs(a + a) ** 2
        yield allclose, B.ew_sums(f(a), g(b)), np.abs(a + b)
        yield allclose, B.ew_sums(f(a)), np.abs(a + a)
