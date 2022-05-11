import numpy as np
import pytest

import lab as B

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
                None,
                (0, 1, 2),
                (0, 2, 1),
                (1, 0, 2),
                (1, 2, 0),
                (2, 1, 0),
                (2, 0, 1),
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


def test_einsum(check_lazy_shapes):
    for eq in ["ij,ij->", "ij,jk->ik", "ii,ii->"]:
        check_function(B.einsum, (Value(eq), Tensor(3, 3), Tensor(3, 3)))
    for eq in ["...ij,...ij->...", "...ij,...jk->...ik", "...ii,...ii->..."]:
        check_function(B.einsum, (Value(eq), Tensor(4, 3, 3), Tensor(4, 3, 3)))


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
    # Cannot test tensors of rank higher than two, because TensorFlows broadcasting
    # behaviour does not allow that.

    # Test full Kronecker product.
    check_function(B.kron, (Tensor(2, 3), Tensor(4, 5)))
    check_function(lambda x, y: B.kron(x, y, 0, 1), (Tensor(2, 3), Tensor(4, 5)))
    check_function(lambda x, y: B.kron(x, y, -2, -1), (Tensor(2, 3), Tensor(4, 5)))

    # Test Kronecker product over dimension 0.
    check_function(lambda x, y: B.kron(x, y, 0), (Tensor(2, 3), Tensor(4, 3)))
    check_function(lambda x, y: B.kron(x, y, -2), (Tensor(2, 3), Tensor(4, 3)))

    # Test Kronecker product over dimension 1.
    check_function(lambda x, y: B.kron(x, y, 1), (Tensor(2, 3), Tensor(2, 5)))
    check_function(lambda x, y: B.kron(x, y, -1), (Tensor(2, 3), Tensor(2, 5)))

    # Test checking of shapes.
    with pytest.raises(ValueError):
        B.kron(Tensor(2).np(), Tensor(4, 5).np())
    with pytest.raises(ValueError):
        B.kron(Tensor(2, 3).np(), Tensor(4, 5).np(), 1)


def test_svd(check_lazy_shapes):
    # Take absolute value because the sign of the result is undetermined.
    def svd(a, compute_uv=True):
        if compute_uv:
            u, s, v = B.svd(a, compute_uv=True)
            return B.abs(u), s, B.abs(v)
        else:
            return B.svd(a, compute_uv=False)

    check_function(svd, (Tensor(3, 2),), {"compute_uv": Bool()})
    check_function(svd, (Tensor(4, 3, 2),), {"compute_uv": Bool()})


def test_eig(check_lazy_shapes):
    # Order of eigenvalues and signs of eigenvectors may be different.
    def compute_order(vals):
        key = B.imag(vals) + B.real(vals)
        return B.argsort(key)

    def eig(a, compute_eigvecs=True):
        if compute_eigvecs:
            vals, vecs = B.eig(a, compute_eigvecs=True)
            vals = B.flatten(vals)
            if B.rank(vecs) == 3:
                vecs = B.transpose(vecs, perm=(1, 0, 2))
                vecs = B.reshape(vecs, 3, -1)
            order = compute_order(vals)
            return B.take(vals, order), B.abs(B.take(vecs, order, axis=1))
        else:
            vals = B.flatten(B.eig(a, compute_eigvecs=False))
            return B.take(vals, compute_order(vals))

    # Some frameworks convert eigenvalues to real if the imaginary parts are all zero
    # exactly.
    check_function(
        eig,
        (Tensor(3, 3),),
        {"compute_eigvecs": Bool()},
        assert_dtype=False,
        skip=[B.AGNumeric],
    )
    check_function(
        eig,
        (Tensor(4, 3, 3),),
        {"compute_eigvecs": Bool()},
        assert_dtype=False,
        skip=[B.AGNumeric],
    )


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


def test_cholesky_retry_factor(check_lazy_shapes):
    # Try `cholesky_retry_factor = 1`.
    B.cholesky_retry_factor = 1
    B.cholesky(B.zeros(3, 3))
    B.cholesky(B.zeros(3, 3) - 0.5 * B.eye(3) * B.epsilon)
    with pytest.raises(np.linalg.LinAlgError):
        B.cholesky(B.zeros(3, 3) - 0.5 * B.eye(3) * 10 * B.epsilon)
    with pytest.raises(np.linalg.LinAlgError):
        B.cholesky(B.zeros(3, 3) - 0.5 * B.eye(3) * 100 * B.epsilon)

    # Try `cholesky_retry_factor = 10`.
    B.cholesky_retry_factor = 10
    B.cholesky(B.zeros(3, 3))
    B.cholesky(B.zeros(3, 3) - 0.5 * B.eye(3) * B.epsilon)
    B.cholesky(B.zeros(3, 3) - 0.5 * B.eye(3) * 10 * B.epsilon)
    with pytest.raises(np.linalg.LinAlgError):
        B.cholesky(B.zeros(3, 3) - 0.5 * B.eye(3) * 100 * B.epsilon)

    # Try `cholesky_retry_factor = 100`.
    B.cholesky_retry_factor = 100
    B.cholesky(B.zeros(3, 3))
    B.cholesky(B.zeros(3, 3) - 0.5 * B.eye(3) * B.epsilon)
    B.cholesky(B.zeros(3, 3) - 0.5 * B.eye(3) * 10 * B.epsilon)
    B.cholesky(B.zeros(3, 3) - 0.5 * B.eye(3) * 100 * B.epsilon)

    # Reset the factor!
    B.cholesky_retry_factor = 1


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


@pytest.mark.parametrize("batch_a", [False, True])
@pytest.mark.parametrize("batch_b", [False, True])
def test_pw_2d(check_lazy_shapes, batch_a, batch_b):
    a = Tensor(*((3,) if batch_a else ()), 5, 2).np()
    b = Tensor(*((3,) if batch_b else ()), 10, 2).np()

    # In this case, allow for 1e-7 absolute error, because the computation is
    # approximate.
    def _approx(a, b):
        approx(a, b, atol=1e-7)

    dists2_ab = np.zeros((*((3,) if batch_a or batch_b else ()), 5, 10))
    dists2_aa = np.zeros((*((3,) if batch_a else ()), 5, 5))
    sums2_ab = np.zeros((*((3,) if batch_a or batch_b else ()), 5, 10))
    sums2_aa = np.zeros((*((3,) if batch_a else ()), 5, 5))
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

    _approx(B.pw_dists2(a, b), dists2_ab)
    _approx(B.pw_dists2(a), dists2_aa)
    _approx(B.pw_dists(a, b), np.maximum(dists2_ab, 1e-30) ** 0.5)
    _approx(B.pw_dists(a), np.maximum(dists2_aa, 1e-30) ** 0.5)
    _approx(B.pw_sums2(a, b), sums2_ab)
    _approx(B.pw_sums2(a), sums2_aa)
    _approx(B.pw_sums(a, b), np.maximum(sums2_ab, 1e-30) ** 0.5)
    _approx(B.pw_sums(a), np.maximum(sums2_aa, 1e-30) ** 0.5)


@pytest.mark.parametrize("batch_a", [False, True])
@pytest.mark.parametrize("batch_b", [False, True])
def test_pw_1d(check_lazy_shapes, batch_a, batch_b):
    a = Tensor(*((3,) if batch_a else ()), 5, 1).np()
    b = Tensor(*((3,) if batch_b else ()), 10, 1).np()

    # Check that we can feed both rank 1 and rank 2 tensors.
    for squeeze_a in [True, False]:
        for squeeze_b in [True, False]:
            if squeeze_a and B.rank(a) == 2:
                a2 = a[..., 0]
            else:
                a2 = a
            if squeeze_b and B.rank(b) == 2:
                b2 = b[..., 0]
            else:
                b2 = b

            approx(B.pw_dists2(a2, b2), np.abs(a - B.t(b)) ** 2)
            approx(B.pw_dists2(a2), np.abs(a - B.t(a)) ** 2)
            approx(B.pw_dists(a2, b2), np.abs(a - B.t(b)))
            approx(B.pw_dists(a2), np.abs(a - B.t(a)))
            approx(B.pw_sums2(a2, b2), np.abs(a + B.t(b)) ** 2)
            approx(B.pw_sums2(a2), np.abs(a + B.t(a)) ** 2)
            approx(B.pw_sums(a2, b2), np.abs(a + B.t(b)))
            approx(B.pw_sums(a2), np.abs(a + B.t(a)))


@pytest.mark.parametrize("batch_a", [False, True])
@pytest.mark.parametrize("batch_b", [False, True])
def test_ew_2d(check_lazy_shapes, batch_a, batch_b):
    a = Tensor(*((3,) if batch_a else ()), 10, 2).np()
    b = Tensor(*((3,) if batch_b else ()), 10, 2).np()

    dists2_ab = np.zeros((*((3,) if batch_a or batch_b else ()), 10, 1))
    dists2_aa = np.zeros((*((3,) if batch_a else ()), 10, 1))
    sums2_ab = np.zeros((*((3,) if batch_a or batch_b else ()), 10, 1))
    sums2_aa = np.zeros((*((3,) if batch_a else ()), 10, 1))
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


@pytest.mark.parametrize("batch_a", [False, True])
@pytest.mark.parametrize("batch_b", [False, True])
def test_ew_1d(check_lazy_shapes, batch_a, batch_b):
    a = Tensor(*((3,) if batch_a else ()), 10, 1).np()
    b = Tensor(*((3,) if batch_b else ()), 10, 1).np()

    # Check that we can feed both rank 1 and rank 2 tensors.
    for squeeze_a in [True, False]:
        for squeeze_b in [True, False]:
            if squeeze_a and B.rank(a) == 2:
                a2 = a[..., 0]
            else:
                a2 = a
            if squeeze_b and B.rank(b) == 2:
                b2 = b[..., 0]
            else:
                b2 = b

            approx(B.ew_dists2(a2, b2), np.abs(a - b) ** 2)
            approx(B.ew_dists2(a2), np.zeros((*((3,) if batch_a else ()), 10, 1)))
            approx(B.ew_dists(a2, b2), np.abs(a - b))
            approx(B.ew_dists(a2), np.zeros((*((3,) if batch_a else ()), 10, 1)))
            approx(B.ew_sums2(a2, b2), np.abs(a + b) ** 2)
            approx(B.ew_sums2(a2), np.abs(a + a) ** 2)
            approx(B.ew_sums(a2, b2), np.abs(a + b))
            approx(B.ew_sums(a2), np.abs(a + a))
