import logging

from . import dispatch, B
from .types import Numeric
from .util import abstract

__all__ = [
    "epsilon",
    "transpose",
    "t",
    "T",
    "matmul",
    "mm",
    "dot",
    "kron",
    "trace",
    "svd",
    "solve",
    "inv",
    "det",
    "logdet",
    "expm",
    "logm",
    "cholesky",
    "chol",
    "cholesky_solve",
    "cholsolve",
    "triangular_solve",
    "trisolve",
    "toeplitz_solve",
    "toepsolve",
    "outer",
    "reg",
    "pw_dists2",
    "pw_dists",
    "ew_dists2",
    "ew_dists",
    "pw_sums2",
    "pw_sums",
    "ew_sums2",
    "ew_sums",
]

log = logging.getLogger(__name__)

epsilon = 1e-12  #: Magnitude of diagonal to regularise matrices with.


def _default_perm(a):
    rank_a = B.rank(a)
    perm = list(range(rank_a))

    # Switch the last two dimensions if `rank_a >= 2`.
    if len(perm) >= 2:
        perm[-2], perm[-1] = perm[-1], perm[-2]

    return perm


@dispatch
@abstract()
def transpose(a: Numeric, perm=None):  # pragma: no cover
    """Transpose a matrix.

    Args:
        a (tensor): Matrix to transpose.
        perm (list[int] or tuple[int], optional): Permutation. Defaults to
            switching the last two axes.

    Returns:
        tensor: Transposition of `a`.
    """


t = transpose  #: Shorthand for `transpose`.
T = transpose  #: Shorthand for `transpose`.


@dispatch
@abstract(promote=2)
def matmul(a, b, tr_a=False, tr_b=False):  # pragma: no cover
    """Matrix multiplication.

    Args:
        a (tensor): First matrix.
        b (tensor): Second matrix.
        tr_a (bool, optional): Transpose first matrix. Defaults to `False`.
        tr_b (bool, optional): Transpose second matrix. Defaults to `False`.

    Returns:
        tensor: Matrix product of `a` and `b`.
    """


mm = matmul  #: Shorthand for `matmul`.
dot = matmul  #: Shorthand for `matmul`.


@dispatch
@abstract()
def trace(a: Numeric, axis1=0, axis2=1):  # pragma: no cover
    """Compute the trace of a tensor.

    Args:
        a (tensor): Tensor to compute trace of.
        axis1 (int, optional): First dimension to compute trace over. Defaults
            to `0`.
        axis2 (int, optional): Second dimension to compute trace over. Defaults
            to `1`.

    Returns:
        tensor: Trace.
    """


@dispatch
@abstract(promote=2)
def kron(a, b):  # pragma: no cover
    """Kronecker product.

    Args:
        a (tensor): First matrix.
        b (tensor): Second matrix.

    Returns:
        tensor: Kronecker product of `a` and `b`.
    """


@dispatch
@abstract()
def svd(a: Numeric, compute_uv=True):  # pragma: no cover
    """Compute the singular value decomposition.

    Note:
        PyTorch does not allow batch computation.

    Args:
        a (tensor): Matrix to decompose.
        compute_uv (bool, optional): Also compute `U` and `V`. Defaults to
            `True`.

    Returns:
        tuple: `(U, S, V)` is `compute_uv` is `True` and just `S` otherwise.
    """


@dispatch
@abstract(promote=2)
def solve(a, b):  # pragma: no cover
    """Solve the linear system `a x = b`.

    Args:
        a (tensor): LHS `a`.
        b (tensor): RHS `b`.

    Returns:
        tensor: Solution `x`.
    """


@dispatch
@abstract()
def inv(a):  # pragma: no cover
    """Compute the inverse of `a`.

    Args:
        a (tensor): Matrix to compute inverse of.

    Returns:
        tensor: Inverse of `a`.
    """


@dispatch
@abstract()
def det(a):  # pragma: no cover
    """Compute the determinant of `a`.

    Args:
        a (tensor): Matrix to compute determinant of.

    Returns:
        scalar: Determinant of `a`
    """


@dispatch
@abstract()
def logdet(a):  # pragma: no cover
    """Compute the log-determinant of `a`.

    Args:
        a (tensor): Matrix to compute log-determinant of.

    Returns:
        scalar: Log-determinant of `a`
    """


@dispatch
@abstract()
def expm(a):  # pragma: no cover
    """Compute the matrix exponential of `a`.

    Args:
        a (tensor): Matrix to matrix exponential of.

    Returns:
        scalar: Matrix exponential of `a`
    """


@dispatch
@abstract()
def logm(a):  # pragma: no cover
    """Compute the matrix logarithm of `a`.

    Args:
        a (tensor): Matrix to matrix logarithm of.

    Returns:
        scalar: Matrix logarithm of `a`
    """


@dispatch
@abstract()
def cholesky(a: Numeric):  # pragma: no cover
    """Compute the Cholesky decomposition.

    Args:
        a (tensor): Matrix to decompose.

    Returns:
        tensor: Cholesky decomposition.
    """


chol = cholesky  #: Shorthand for `cholesky`.


@dispatch
@abstract(promote=2)
def cholesky_solve(a, b):  # pragma: no cover
    """Solve the linear system `a x = b` given the Cholesky factorisation of
    `a`.

    Args:
        a (tensor): Cholesky factorisation of `a`.
        b (tensor): RHS `b`.

    Returns:
        tensor: Solution `x`.
    """


cholsolve = cholesky_solve  #: Shorthand for `cholesky_solve`.


@dispatch
@abstract(promote=2)
def triangular_solve(a, b, lower_a=True):  # pragma: no cover
    """Solve the linear system `a x = b` where `a` is triangular.

    Args:
        a (tensor): Triangular matrix `a`.
        b (tensor): RHS `b`.
        lower_a (bool, optional): Indicate that `a` is lower triangular
            instead of upper triangular. Defaults to `True`.

    Returns:
        tensor: Solution `x`.
    """


trisolve = triangular_solve  #: Shorthand for `triangular_solve`.


@dispatch
@abstract(promote=3)
def toeplitz_solve(a, b, c):  # pragma: no cover
    """Solve the linear system `toep(a, b) x = c` where `toep(a, b)` is a
    Toeplitz matrix.

    Args:
        a (tensor): First column of the Toeplitz matrix.
        b (tensor, optional): *Except for the first element*, first row of the
            Toeplitz matrix. Defaults to `a[1:]`.
        c (tensor): RHS `c`.

    Returns:
        tensor: Solution `x`.
    """


@dispatch
def toeplitz_solve(a, c):
    return toeplitz_solve(a, a[1:], c)


toepsolve = toeplitz_solve  #: Shorthand for `toeplitz_solve`.


@dispatch
def outer(a, b):
    """Compute the outer product between two vectors or matrices.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor: Outer product of `a` and `b`.
    """
    a, b = _a_b_uprank(a, b)

    # Optimise the case that both are column vectors.
    if B.shape(a)[-1] == 1 and B.shape(b)[-1] == 1:
        return a * B.transpose(b)

    return B.matmul(a, b, tr_b=True)


def _a_b_uprank(a, b):
    # Needs to be at least rank two.
    target_rank = max(B.rank(a), B.rank(b), 2)
    return B.uprank(a, rank=target_rank), B.uprank(b, rank=target_rank)


@dispatch
def outer(a):
    return outer(a, a)


@dispatch
def reg(a, diag=None, clip=True):
    """Add a diagonal to a matrix.

    Args:
        a (matrix): Matrix to add a diagonal to.
        diag (float, optional): Magnitude of the diagonal to add. Defaults to
            `.linear_algebra.epsilon`.
        clip (bool, optional): Let `diag` be at least `.linear_algebra.epsilon`.
            Defaults to `True`.

    Returns:
        matrix: Regularised version of `a`.
    """
    # Careful to use `B.epsilon` here and not `epsilon`! Otherwise, changes
    # will not be tracked.
    if diag is None:
        diag = B.epsilon
    if clip and diag is not B.epsilon:
        diag = B.maximum(diag, B.epsilon)
    return a + diag * B.eye(a)


@dispatch
def pw_dists2(a, b):
    """Compute the square the Euclidean norm of the pairwise differences between two
    matrices where rows correspond to elements and columns to features.

    Args:
        a (matrix): First matrix.
        b (matrix, optional): Second matrix. Defaults to `a`.

    Returns:
        matrix: Square of the Euclidean norm of the pairwise differences
            between the elements of `a` and `b`.
    """
    a, b = _a_b_uprank(a, b)

    # Optimise the one-dimensional case.
    if B.shape(a)[-1] == 1 and B.shape(b)[-1] == 1:
        return (a - B.transpose(b)) ** 2

    norms_a = B.sum(a ** 2, axis=-1)[..., :, None]
    norms_b = B.sum(b ** 2, axis=-1)[..., None, :]
    return norms_a + norms_b - 2 * B.matmul(a, b, tr_b=True)


@dispatch
def pw_dists2(a):
    return pw_dists2(a, a)


@dispatch
def pw_dists(a, b):
    """Compute the Euclidean norm of the pairwise differences between two matrices
    where rows correspond to elements and columns to features.

    Args:
        a (matrix): First matrix.
        b (matrix, optional): Second matrix. Defaults to `a`.

    Returns:
        matrix: Euclidean norm of the pairwise differences between the
            elements of `a` and `b`.
    """
    a, b = _a_b_uprank(a, b)

    # Optimise the one-dimensional case.
    if B.shape(a)[-1] == 1 and B.shape(b)[-1] == 1:
        return B.abs(a - B.transpose(b))

    return B.sqrt(B.maximum(B.pw_dists2(a, b), B.cast(B.dtype(a), 1e-30)))


@dispatch
def pw_dists(a):
    return pw_dists(a, a)


@dispatch
def ew_dists2(a, b):
    """Compute the square the Euclidean norm of the element-wise differences between
    two matrices where rows correspond to elements and columns to features.

    Args:
        a (matrix): First matrix.
        b (matrix, optional): Second matrix. Defaults to `a`.

    Returns:
        matrix: Square of the Euclidean norm of the element-wise differences
            between the elements of `a` and `b`.
    """
    a, b = _a_b_uprank(a, b)
    return B.sum((a - b) ** 2, axis=-1)[..., :, None]


@dispatch
def ew_dists2(a):
    return ew_dists2(a, a)


@dispatch
def ew_dists(a, b):
    """Compute the Euclidean norm of the element-wise differences between two
    matrices where rows correspond to elements and columns to features.

    Args:
        a (matrix): First matrix.
        b (matrix, optional): Second matrix. Defaults to `a`.

    Returns:
        matrix: Euclidean norm of the element-wise differences between the
            elements of `a` and `b`.
    """
    a, b = _a_b_uprank(a, b)

    # Optimise the one-dimensional case.
    if B.shape(a)[-1] == 1 and B.shape(b)[-1] == 1:
        return B.abs(a - b)

    return B.sqrt(B.maximum(B.ew_dists2(a, b), B.cast(B.dtype(a), 1e-30)))


@dispatch
def ew_dists(a):
    return ew_dists(a, a)


@dispatch
def pw_sums2(a, b):
    """Compute the square the Euclidean norm of the pairwise sums between two
    matrices where rows correspond to elements and columns to features.

    Args:
        a (matrix): First matrix.
        b (matrix, optional): Second matrix. Defaults to `a`.

    Returns:
        matrix: Square of the Euclidean norm of the pairwise sums
            between the elements of `a` and `b`.
    """
    a, b = _a_b_uprank(a, b)

    # Optimise the one-dimensional case.
    if B.shape(a)[-1] == 1 and B.shape(b)[-1] == 1:
        return (a + B.transpose(b)) ** 2

    norms_a = B.sum(a ** 2, axis=-1)[..., :, None]
    norms_b = B.sum(b ** 2, axis=-1)[..., None, :]
    return norms_a + norms_b + 2 * B.matmul(a, b, tr_b=True)


@dispatch
def pw_sums2(a):
    return pw_sums2(a, a)


@dispatch
def pw_sums(a, b):
    """Compute the Euclidean norm of the pairwise sums between two
    matrices where rows correspond to elements and columns to features.

    Args:
        a (matrix): First matrix.
        b (matrix, optional): Second matrix. Defaults to `a`.

    Returns:
        matrix: Euclidean norm of the pairwise sums between the
            elements of `a` and `b`.
    """
    a, b = _a_b_uprank(a, b)

    # Optimise the one-dimensional case.
    if B.shape(a)[-1] == 1 and B.shape(b)[-1] == 1:
        return B.abs(a + B.transpose(b))

    return B.sqrt(B.maximum(B.pw_sums2(a, b), B.cast(B.dtype(a), 1e-30)))


@dispatch
def pw_sums(a):
    return pw_sums(a, a)


@dispatch
def ew_sums2(a, b):
    """Compute the square the Euclidean norm of the element-wise
    sums between two matrices where rows correspond to elements and
    columns to features.

    Args:
        a (matrix): First matrix.
        b (matrix, optional): Second matrix. Defaults to `a`.

    Returns:
        matrix: Square of the Euclidean norm of the element-wise sums
            between the elements of `a` and `b`.
    """
    a, b = _a_b_uprank(a, b)
    return B.sum((a + b) ** 2, axis=-1)[..., :, None]


@dispatch
def ew_sums2(a):
    return ew_sums2(a, a)


@dispatch
def ew_sums(a, b):
    """Compute the Euclidean norm of the element-wise sums between two
    matrices where rows correspond to elements and columns to features.

    Args:
        a (matrix): First matrix.
        b (matrix, optional): Second matrix. Defaults to `a`.

    Returns:
        matrix: Euclidean norm of the element-wise sums between the
            elements of `a` and `b`.
    """
    a, b = _a_b_uprank(a, b)

    # Optimise the one-dimensional case.
    if B.shape(a)[-1] == 1 and B.shape(b)[-1] == 1:
        return B.abs(a + b)

    return B.sqrt(B.maximum(B.ew_sums2(a, b), B.cast(B.dtype(a), 1e-30)))


@dispatch
def ew_sums(a):
    return ew_sums(a, a)
