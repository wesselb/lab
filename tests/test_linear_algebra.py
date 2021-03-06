import lab as B
import numpy as np
import pytest
import scipy as sp
from plum import NotFoundLookupError

# noinspection PyUnresolvedReferences
from .util import (
    check_function,
    Tensor,
    Matrix,
    PSD,
    PSDTriangular,
    Value,
    Bool,
    approx,
    check_lazy_shapes,
)


def test_constants(check_lazy_shapes):
    assert B.epsilon == 1e-12


@pytest.mark.parametrize("f", [B.transpose, B.T, B.t])
def test_transpose(f, check_lazy_shapes):
    # Check consistency.
    check_function(f, (Tensor(),))
    check_function(f, (Tensor(2),), {"perm": Value(None, (0,))})
    check_function(f, (Tensor(2, 3),), {"perm": Value(None, (0, 1), (1, 0))})
    check_function(
        f,
        (Tensor(2, 3, 4),),
        {
            "perm": Value(
                None, (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 1, 0), (2, 0, 1)
            )
        },
    )

    # Check correctness of zero-dimensional case.
    assert f(1) is 1

    # Check correctness of one-dimensional case.
    a = Tensor(2).np()
    approx(f(a, perm=None), a[None, :])
    approx(f(a, perm=(0,)), a)

    # Check correctness of three-dimensional case.
    a = Tensor(2, 3, 4).np()
    approx(f(a), np.transpose(a, axes=(0, 2, 1)))
    approx(f(a, perm=(1, 2, 0)), np.transpose(a, axes=(1, 2, 0)))


@pytest.mark.parametrize("f", [B.matmul, B.mm, B.dot])
def test_matmul(f, check_lazy_shapes):
    check_function(f, (Tensor(3, 3), Tensor(3, 3)), {"tr_a": Bool(), "tr_b": Bool()})
    check_function(
        f, (Tensor(4, 3, 3), Tensor(4, 3, 3)), {"tr_a": Bool(), "tr_b": Bool()}
    )


def test_trace(check_lazy_shapes):
    # Check default call.
    check_function(
        B.trace, (Tensor(2, 3, 4, 5),), {"axis1": Value(0), "axis2": Value(1)}
    )

    # Check calls with `axis1 < axis2`.
    check_function(
        B.trace, (Tensor(2, 3, 4, 5),), {"axis1": Value(0, 1), "axis2": Value(2, 3)}
    )

    # Check calls with `axis1 > axis2`.
    check_function(
        B.trace, (Tensor(2, 3, 4, 5),), {"axis1": Value(2, 3), "axis2": Value(0, 1)}
    )

    # Check that call with `axis1 == axis2` raises an error in NumPy.
    with pytest.raises(ValueError):
        B.trace(Matrix().ag(), axis1=0, axis2=0)


def test_kron(check_lazy_shapes):
    check_function(B.kron, (Tensor(2, 3), Tensor(4, 5)))
    # Cannot test tensors of higher rank, because TensorFlows broadcasting
    # behaviour does not allow that.
    with pytest.raises(ValueError):
        B.kron(Tensor(2).tf(), Tensor(4, 5).tf())
    with pytest.raises(ValueError):
        B.kron(Tensor(2).torch(), Tensor(4, 5).torch())


def test_svd(check_lazy_shapes):
    # Take absolute value because the sign of the result is undetermined.
    def svd(a, compute_uv=True):
        if compute_uv:
            u, s, v = B.svd(a, compute_uv=True)
            return B.abs(u), s, B.abs(v)
        else:
            return B.svd(a, compute_uv=False)

    check_function(svd, (Tensor(3, 2),), {"compute_uv": Bool()})
    # Torch does not allow batch computation.


def test_solve(check_lazy_shapes):
    check_function(B.solve, (Matrix(3, 3), Matrix(3, 4)))
    check_function(B.solve, (Matrix(5, 3, 3), Matrix(5, 3, 4)))


def test_inv(check_lazy_shapes):
    check_function(B.inv, (Matrix(),))
    check_function(B.inv, (Matrix(4, 3, 3),))


def test_pinv(check_lazy_shapes):
    a = Matrix(4, 6, 3).np()
    assert B.shape(B.pinv(a)) == (4, 3, 6)
    assert B.shape(B.pinv(B.pinv(a))) == (4, 6, 3)
    approx(a, B.pinv(B.pinv(a)))


def test_det(check_lazy_shapes):
    check_function(B.det, (Matrix(),))
    check_function(B.det, (Matrix(4, 3, 3),))


def test_logdet(check_lazy_shapes):
    check_function(B.logdet, (PSD(),))
    check_function(B.logdet, (PSD(4, 3, 3),))


def test_expm(check_lazy_shapes):
    check_function(B.expm, (Matrix(),))


def test_logm(check_lazy_shapes):
    mat = B.eye(3) + 0.1 * B.randn(3, 3)
    check_function(B.logm, (Tensor(mat=mat),))


@pytest.mark.parametrize("f", [B.cholesky, B.chol])
def test_cholesky(f, check_lazy_shapes):
    check_function(f, (PSD(),))
    check_function(f, (PSD(4, 3, 3),))


@pytest.mark.parametrize("f", [B.cholesky_solve, B.cholsolve])
def test_cholesky_solve(f, check_lazy_shapes):
    check_function(f, (PSDTriangular(3, 3), Matrix(3, 4)))
    check_function(f, (PSDTriangular(5, 3, 3), Matrix(5, 3, 4)))


@pytest.mark.parametrize("f", [B.triangular_solve, B.trisolve])
def test_triangular_solve(f, check_lazy_shapes):
    check_function(f, (PSDTriangular(3, 3), Matrix(3, 4)), {"lower_a": Value(True)})
    check_function(
        f, (PSDTriangular(5, 3, 3), Matrix(5, 3, 4)), {"lower_a": Value(True)}
    )
    check_function(
        f, (PSDTriangular(3, 3, upper=True), Matrix(3, 4)), {"lower_a": Value(False)}
    )
    check_function(
        f,
        (PSDTriangular(5, 3, 3, upper=True), Matrix(5, 3, 4)),
        {"lower_a": Value(False)},
    )


@pytest.mark.parametrize("f", [B.toeplitz_solve, B.toepsolve])
def test_toeplitz_solve(f, check_lazy_shapes):
    check_function(f, (Tensor(3), Tensor(2), Matrix(3, 4)))
    check_function(f, (Tensor(3), Matrix(3, 4)))


@pytest.mark.parametrize("shape", [(5,), (5, 1), (5, 2), (3, 5, 1), (3, 5, 2)])
def test_outer(shape, check_lazy_shapes):
    a = Tensor(*shape).np()
    b = Tensor(*shape).np()
    approx(B.outer(a, b), B.matmul(B.uprank(a), B.uprank(b), tr_b=True))
    approx(B.outer(a), B.outer(a, a))
    approx(B.outer(b), B.outer(b, b))


def test_reg(check_lazy_shapes):
    old_epsilon = B.epsilon
    B.epsilon = 10
    a = Matrix(2, 3).np()
    approx(B.reg(a, diag=None, clip=False), a + 10 * np.eye(*a.shape))
    approx(B.reg(a, diag=None, clip=True), a + 10 * np.eye(*a.shape))
    approx(B.reg(a, diag=1, clip=False), a + 1 * np.eye(*a.shape))
    approx(B.reg(a, diag=1, clip=True), a + 10 * np.eye(*a.shape))
    approx(B.reg(a, diag=100, clip=False), a + 100 * np.eye(*a.shape))
    approx(B.reg(a, diag=100, clip=True), a + 100 * np.eye(*a.shape))
    B.epsilon = old_epsilon


@pytest.mark.parametrize("batch_shape", [(), (3,)])
def test_pw_2d(check_lazy_shapes, batch_shape):
    # In this case, allow for 1e-7 absolute error, because the computation is
    # approximate.
    def approx_allclose(a, b):
        approx(a, b, atol=1e-7)

    a, b = Tensor(*batch_shape, 5, 2).np(), Tensor(*batch_shape, 10, 2).np()
    dists2_ab = np.zeros((*batch_shape, 5, 10))
    dists2_aa = np.zeros((*batch_shape, 5, 5))
    sums2_ab = np.zeros((*batch_shape, 5, 10))
    sums2_aa = np.zeros((*batch_shape, 5, 5))
    for i in range(5):
        for j in range(10):
            dists2_ab[..., i, j] = np.sum((a[..., i, :] - b[..., j, :]) ** 2, axis=-1)
            sums2_ab[..., i, j] = np.sum((a[..., i, :] + b[..., j, :]) ** 2, axis=-1)
            if j < 5:
                dists2_aa[..., i, j] = np.sum(
                    (a[..., i, :] - a[..., j, :]) ** 2, axis=-1
                )
                sums2_aa[..., i, j] = np.sum(
                    (a[..., i, :] + a[..., j, :]) ** 2, axis=-1
                )

    approx_allclose(B.pw_dists2(a, b), dists2_ab)
    approx_allclose(B.pw_dists2(a), dists2_aa)
    approx_allclose(B.pw_dists(a, b), np.maximum(dists2_ab, 1e-30) ** 0.5)
    approx_allclose(B.pw_dists(a), np.maximum(dists2_aa, 1e-30) ** 0.5)
    approx_allclose(B.pw_sums2(a, b), sums2_ab)
    approx_allclose(B.pw_sums2(a), sums2_aa)
    approx_allclose(B.pw_sums(a, b), np.maximum(sums2_ab, 1e-30) ** 0.5)
    approx_allclose(B.pw_sums(a), np.maximum(sums2_aa, 1e-30) ** 0.5)


@pytest.mark.parametrize("batch_shape", [(), (3,)])
def test_pw_1d(check_lazy_shapes, batch_shape):
    a, b = Tensor(*batch_shape, 5, 1).np(), Tensor(*batch_shape, 10, 1).np()

    # Check that we can feed both rank 1 and rank 2 tensors.
    for squeeze_a in [True, False]:
        for squeeze_b in [True, False]:
            # One-argument case:
            if B.rank(a) == 2 and squeeze_a:
                a1 = a[..., 0]
            else:
                a1 = a

            # Two-argument case:
            if squeeze_a and squeeze_b:
                if max(B.rank(a), B.rank(b)) > 2:
                    # Don't squeeze in this case. That will compute the wrong thing.
                    a2, b2 = a, b
                else:
                    a2, b2 = a[..., 0], b[..., 0]
            elif squeeze_a:
                a2, b2 = a[..., 0], b
            elif squeeze_b:
                a2, b2 = a, b[..., 0]
            else:
                a2, b2 = a, b

            approx(B.pw_dists2(a2, b2), np.abs(a - B.t(b)) ** 2)
            approx(B.pw_dists2(a1), np.abs(a - B.t(a)) ** 2)
            approx(B.pw_dists(a2, b2), np.abs(a - B.t(b)))
            approx(B.pw_dists(a1), np.abs(a - B.t(a)))
            approx(B.pw_sums2(a2, b2), np.abs(a + B.t(b)) ** 2)
            approx(B.pw_sums2(a1), np.abs(a + B.t(a)) ** 2)
            approx(B.pw_sums(a2, b2), np.abs(a + B.t(b)))
            approx(B.pw_sums(a1), np.abs(a + B.t(a)))


@pytest.mark.parametrize("batch_shape", [(), (3,)])
def test_ew_2d(check_lazy_shapes, batch_shape):
    a, b = Tensor(*batch_shape, 10, 2).np(), Tensor(*batch_shape, 10, 2).np()
    dists2_ab = np.zeros((*batch_shape, 10, 1))
    dists2_aa = np.zeros((*batch_shape, 10, 1))
    sums2_ab = np.zeros((*batch_shape, 10, 1))
    sums2_aa = np.zeros((*batch_shape, 10, 1))
    for i in range(10):
        dists2_ab[..., i, 0] = np.sum((a[..., i, :] - b[..., i, :]) ** 2, axis=-1)
        dists2_aa[..., i, 0] = np.sum((a[..., i, :] - a[..., i, :]) ** 2, axis=-1)
        sums2_ab[..., i, 0] = np.sum((a[..., i, :] + b[..., i, :]) ** 2, axis=-1)
        sums2_aa[..., i, 0] = np.sum((a[..., i, :] + a[..., i, :]) ** 2, axis=-1)

    approx(B.ew_dists2(a, b), dists2_ab)
    approx(B.ew_dists2(a), dists2_aa)
    approx(B.ew_dists(a, b), np.maximum(dists2_ab, 1e-30) ** 0.5)
    approx(B.ew_dists(a), np.maximum(dists2_aa, 1e-30) ** 0.5)
    approx(B.ew_sums2(a, b), sums2_ab)
    approx(B.ew_sums2(a), sums2_aa)
    approx(B.ew_sums(a, b), np.maximum(sums2_ab, 1e-30) ** 0.5)
    approx(B.ew_sums(a), np.maximum(sums2_aa, 1e-30) ** 0.5)


@pytest.mark.parametrize("batch_shape", [(), (3,)])
def test_ew_1d(check_lazy_shapes, batch_shape):
    a, b = Tensor(*batch_shape, 10, 1).np(), Tensor(*batch_shape, 10, 1).np()

    # Check that we can feed both rank 1 and rank 2 tensors.
    for squeeze_a in [True, False]:
        for squeeze_b in [True, False]:
            # One-argument case:
            if B.rank(a) == 2 and squeeze_a:
                a1 = a[..., 0]
            else:
                a1 = a

            # Two-argument case:
            if squeeze_a and squeeze_b:
                if max(B.rank(a), B.rank(b)) > 2:
                    # Don't squeeze in this case. That will compute the wrong thing.
                    a2, b2 = a, b
                else:
                    a2, b2 = a[..., 0], b[..., 0]
            elif squeeze_a:
                a2, b2 = a[..., 0], b
            elif squeeze_b:
                a2, b2 = a, b[..., 0]
            else:
                a2, b2 = a, b

            approx(B.ew_dists2(a2, b2), np.abs(a - b) ** 2)
            approx(B.ew_dists2(a1), np.zeros((*batch_shape, 10, 1)))
            approx(B.ew_dists(a2, b2), np.abs(a - b))
            approx(B.ew_dists(a1), np.zeros((*batch_shape, 10, 1)))
            approx(B.ew_sums2(a2, b2), np.abs(a + b) ** 2)
            approx(B.ew_sums2(a1), np.abs(a + a) ** 2)
            approx(B.ew_sums(a2, b2), np.abs(a + b))
            approx(B.ew_sums(a1), np.abs(a + a))


def test_block_diag(check_lazy_shapes):
    # Check that arguments must be given.
    with pytest.raises(NotFoundLookupError):
        B.block_diag()

    elements = [
        B.randn(1, 1),
        B.randn(1, 2),
        B.randn(2, 1),
        B.randn(2, 2),
        B.randn(2, 3),
        B.randn(3, 2),
    ]
    for i in range(1, len(elements) + 1):
        approx(B.block_diag(*elements[:i]), sp.linalg.block_diag(*elements[:i]))
