from itertools import product

import numpy as np
import pytest

import lab as B
from .util import (
    check_function,
    Tensor,
    Matrix,
    PSD,
    PSDTriangular,
    Value,
    Bool,
    allclose
)


def test_constants():
    assert B.epsilon == 1e-12


@pytest.mark.parametrize('f', [B.transpose, B.T, B.t])
def test_transpose(f):
    # Check consistency.
    check_function(f, (Tensor(),))
    check_function(f, (Tensor(2),), {'perm': Value(None, (0,))})
    check_function(f, (Tensor(2, 3),),
                   {'perm': Value(None, (0, 1), (1, 0))})
    check_function(f, (Tensor(2, 3, 4),), {'perm': Value(None,
                                                         (0, 1, 2),
                                                         (0, 2, 1),
                                                         (1, 0, 2),
                                                         (1, 2, 0),
                                                         (2, 1, 0),
                                                         (2, 0, 1))})

    # Check correctness of zero-dimensional case.
    assert f(1) is 1

    # Check correctness of one-dimensional case.
    a = Tensor(2).np()
    allclose(f(a, perm=None), a[None, :])
    allclose(f(a, perm=(0,)), a)

    # Check correctness of three-dimensional case.
    a = Tensor(2, 3, 4).np()
    allclose(f(a), np.transpose(a, axes=(0, 2, 1)))
    allclose(f(a, perm=(1, 2, 0)), np.transpose(a, axes=(1, 2, 0)))


@pytest.mark.parametrize('f', [B.matmul, B.mm, B.dot])
def test_matmul(f):
    check_function(f, (Tensor(3, 3), Tensor(3, 3)),
                   {'tr_a': Bool(), 'tr_b': Bool()})
    check_function(f, (Tensor(4, 3, 3), Tensor(4, 3, 3)),
                   {'tr_a': Bool(), 'tr_b': Bool()})


def test_trace():
    # Check default call.
    check_function(B.trace, (Tensor(2, 3, 4, 5),),
                   {'axis1': Value(0), 'axis2': Value(1)})

    # Check calls with `axis1 < axis2`.
    check_function(B.trace, (Tensor(2, 3, 4, 5),),
                   {'axis1': Value(0, 1), 'axis2': Value(2, 3)})

    # Check calls with `axis1 > axis2`.
    check_function(B.trace, (Tensor(2, 3, 4, 5),),
                   {'axis1': Value(2, 3), 'axis2': Value(0, 1)})

    # Check that call with `axis1 == axis2` raises an error in NumPy.
    with pytest.raises(ValueError):
        B.trace(Matrix().ag(), axis1=0, axis2=0)


def test_kron():
    check_function(B.kron, (Tensor(2, 3), Tensor(4, 5)))
    # Cannot test tensors of higher rank, because TensorFlows broadcasting
    # behaviour does not allow that.
    with pytest.raises(ValueError):
        B.kron(Tensor(2).tf(), Tensor(4, 5).tf())
    with pytest.raises(ValueError):
        B.kron(Tensor(2).torch(), Tensor(4, 5).torch())


def test_svd():
    # Take absolute value because the sign of the result is undetermined.
    def svd(a, compute_uv=True):
        if compute_uv:
            u, s, v = B.svd(a, compute_uv=True)
            return B.abs(u), s, B.abs(v)
        else:
            return B.svd(a, compute_uv=False)

    check_function(svd, (Tensor(3, 2),), {'compute_uv': Bool()})
    # Torch does not allow batch computation.


def test_solve():
    check_function(B.solve, (Matrix(3, 3), Matrix(3, 4)))
    check_function(B.solve, (Matrix(5, 3, 3), Matrix(5, 3, 4)))


def test_inv():
    check_function(B.inv, (Matrix(),))
    check_function(B.inv, (Matrix(4, 3, 3),))


def test_det():
    check_function(B.det, (Matrix(),))
    check_function(B.det, (Matrix(4, 3, 3),))


def test_logdet():
    check_function(B.logdet, (PSD(),))
    check_function(B.logdet, (PSD(4, 3, 3),))


@pytest.mark.parametrize('f', [B.cholesky, B.chol])
def test_cholesky(f):
    check_function(f, (PSD(),))
    check_function(f, (PSD(4, 3, 3),))


@pytest.mark.parametrize('f', [B.cholesky_solve, B.cholsolve])
def test_cholesky_solve(f):
    check_function(f, (PSDTriangular(3, 3), Matrix(3, 4)))
    check_function(f, (PSDTriangular(5, 3, 3), Matrix(5, 3, 4)))


@pytest.mark.parametrize('f', [B.triangular_solve, B.trisolve])
def test_triangular_solve(f):
    check_function(f, (PSDTriangular(3, 3), Matrix(3, 4)),
                   {'lower_a': Value(True)})
    check_function(f, (PSDTriangular(5, 3, 3), Matrix(5, 3, 4)),
                   {'lower_a': Value(True)})
    check_function(f, (PSDTriangular(3, 3, upper=True), Matrix(3, 4)),
                   {'lower_a': Value(False)})
    check_function(f, (PSDTriangular(5, 3, 3, upper=True), Matrix(5, 3, 4)),
                   {'lower_a': Value(False)})


@pytest.mark.parametrize('f', [B.toeplitz_solve, B.toepsolve])
def test_toeplitz_solve(f):
    check_function(f, (Tensor(3), Tensor(2), Matrix(3, 4)))
    check_function(f, (Tensor(3), Matrix(3, 4)))


@pytest.mark.parametrize('a', [Tensor(5).np(), Tensor(5, 1).np()])
@pytest.mark.parametrize('b', [Tensor(5).np(), Tensor(5, 1).np()])
def test_outer(a, b):
    allclose(B.outer(a, b), B.matmul(B.uprank(a), B.uprank(b), tr_b=True))
    allclose(B.outer(a), B.outer(a, a))
    allclose(B.outer(b), B.outer(b, b))


def test_outer_high_rank():
    a = Tensor(5, 3).np()
    b = Tensor(5, 3).np()
    allclose(B.outer(a), B.outer(a, a))
    allclose(B.outer(b), B.outer(b, b))


def test_reg():
    old_epsilon = B.epsilon
    B.epsilon = 10
    a = Matrix(2, 3).np()
    allclose(B.reg(a, diag=None, clip=False), a + 10 * np.eye(*a.shape))
    allclose(B.reg(a, diag=None, clip=True), a + 10 * np.eye(*a.shape))
    allclose(B.reg(a, diag=1, clip=False), a + 1 * np.eye(*a.shape))
    allclose(B.reg(a, diag=1, clip=True), a + 10 * np.eye(*a.shape))
    allclose(B.reg(a, diag=100, clip=False), a + 100 * np.eye(*a.shape))
    allclose(B.reg(a, diag=100, clip=True), a + 100 * np.eye(*a.shape))
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

    approx_allclose(B.pw_dists2(a, b), dists2_ab)
    approx_allclose(B.pw_dists2(a), dists2_aa)
    approx_allclose(B.pw_dists(a, b), np.maximum(dists2_ab, 1e-30) ** .5)
    approx_allclose(B.pw_dists(a), np.maximum(dists2_aa, 1e-30) ** .5)
    approx_allclose(B.pw_sums2(a, b), sums2_ab)
    approx_allclose(B.pw_sums2(a), sums2_aa)
    approx_allclose(B.pw_sums(a, b), np.maximum(sums2_ab, 1e-30) ** .5)
    approx_allclose(B.pw_sums(a), np.maximum(sums2_aa, 1e-30) ** .5)


def test_pw_1d():
    a, b = Matrix(5, 1).np(), Matrix(10, 1).np()

    # Check that we can feed both rank 1 and rank 2 tensors.
    for f, g in product(*([[lambda x: x, lambda x: x[:, 0]]] * 2)):
        allclose(B.pw_dists2(f(a), g(b)), np.abs(a - b.T) ** 2)
        allclose(B.pw_dists2(f(a)), np.abs(a - a.T) ** 2)
        allclose(B.pw_dists(f(a), g(b)), np.abs(a - b.T))
        allclose(B.pw_dists(f(a)), np.abs(a - a.T))
        allclose(B.pw_sums2(f(a), g(b)), np.abs(a + b.T) ** 2)
        allclose(B.pw_sums2(f(a)), np.abs(a + a.T) ** 2)
        allclose(B.pw_sums(f(a), g(b)), np.abs(a + b.T))
        allclose(B.pw_sums(f(a)), np.abs(a + a.T))


def test_ew_2d():
    a, b = Matrix(10, 2).np(), Matrix(10, 2).np()
    dists2_ab, dists2_aa = np.zeros((10, 1)), np.zeros((10, 1))
    sums2_ab, sums2_aa = np.zeros((10, 1)), np.zeros((10, 1))
    for i in range(10):
        dists2_ab[i, 0] = np.sum((a[i] - b[i]) ** 2)
        dists2_aa[i, 0] = np.sum((a[i] - a[i]) ** 2)
        sums2_ab[i, 0] = np.sum((a[i] + b[i]) ** 2)
        sums2_aa[i, 0] = np.sum((a[i] + a[i]) ** 2)

    allclose(B.ew_dists2(a, b), dists2_ab)
    allclose(B.ew_dists2(a), dists2_aa)
    allclose(B.ew_dists(a, b), np.maximum(dists2_ab, 1e-30) ** .5)
    allclose(B.ew_dists(a), np.maximum(dists2_aa, 1e-30) ** .5)
    allclose(B.ew_sums2(a, b), sums2_ab)
    allclose(B.ew_sums2(a), sums2_aa)
    allclose(B.ew_sums(a, b), np.maximum(sums2_ab, 1e-30) ** .5)
    allclose(B.ew_sums(a), np.maximum(sums2_aa, 1e-30) ** .5)


def test_ew_1d():
    a, b = Matrix(10, 1).np(), Matrix(10, 1).np()

    # Check that we can feed both rank 1 and rank 2 tensors.
    for f, g in product(*([[lambda x: x, lambda x: x[:, 0]]] * 2)):
        allclose(B.ew_dists2(f(a), g(b)), np.abs(a - b) ** 2)
        allclose(B.ew_dists2(f(a)), np.zeros((10, 1)))
        allclose(B.ew_dists(f(a), g(b)), np.abs(a - b))
        allclose(B.ew_dists(f(a)), np.zeros((10, 1)))
        allclose(B.ew_sums2(f(a), g(b)), np.abs(a + b) ** 2)
        allclose(B.ew_sums2(f(a)), np.abs(a + a) ** 2)
        allclose(B.ew_sums(f(a), g(b)), np.abs(a + b))
        allclose(B.ew_sums(f(a)), np.abs(a + a))
