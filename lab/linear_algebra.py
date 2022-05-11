import logging
import warnings
from typing import Optional, Union

from . import dispatch, B
from .types import Numeric, Int
from .util import abstract

__all__ = [
    "epsilon",
    "transpose",
    "t",
    "T",
    "matmul",
    "mm",
    "dot",
    "einsum",
    "kron",
    "trace",
    "svd",
    "eig",
    "solve",
    "inv",
    "pinv",
    "det",
    "logdet",
    "expm",
    "logm",
    "cholesky_retry_factor",
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
def transpose(
    a: Numeric, perm: Optional[Union[tuple, list]] = None
):  # pragma: no cover
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
def matmul(a, b, tr_a: bool = False, tr_b: bool = False):  # pragma: no cover
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
@abstract(promote_from=1)
def einsum(equation: str, *elements: Numeric):  # pragma: no cover
    """Tensor contraction via Einstein summation.

    Args:
        equation (str): Equation.
        *elements (tensor): Tensors to contract.

    Returns:
        tensor: Contraction.
    """


@dispatch
@abstract()
def trace(a: Numeric, axis1: Int = -2, axis2: Int = -1):  # pragma: no cover
    """Compute the trace of a tensor.

    Args:
        a (tensor): Tensor to compute trace of.
        axis1 (int, optional): First dimension to compute trace over. Defaults
            to `-2`.
        axis2 (int, optional): Second dimension to compute trace over. Defaults
            to `-1`.

    Returns:
        tensor: Trace.
    """


@dispatch
@abstract(promote=2)
def kron(a, b, *indices: Int):
    """Kronecker product.

    Args:
        a (tensor): First matrix.
        b (tensor): Second matrix.
        *indices (int): Indices to compute the Kronecker product over. Defaults to all
            indices.


    Returns:
        tensor: Kronecker product of `a` and `b`.
    """


@dispatch
def kron(a: Numeric, b: Numeric, *indices: Int):
    a_shape = B.shape(a)
    b_shape = B.shape(b)
    if len(a_shape) != len(b_shape):
        raise ValueError(
            "Can only compute Kronecker products between tensors of equal ranks."
        )

    # Default to computing the Kronecker product over all indices.
    if indices == ():
        indices = range(len(a_shape))
    else:
        # Ensure that all indices are positive indices. Otherwise, the `i in indices`
        # below will fail.
        indices = [len(a_shape) + i if i < 0 else i for i in indices]

    a_indices = ()
    b_indices = ()
    target_shape = ()
    for i in range(len(a_shape)):
        if i in indices:
            a_indices += (slice(None), None)
            b_indices += (None, slice(None))
            target_shape += (a_shape[i] * b_shape[i],)
        else:
            a_indices += (slice(None),)
            b_indices += (slice(None),)
            if a_shape[i] == b_shape[i]:
                target_shape += (a_shape[i],)
            else:
                raise ValueError(
                    f"Shape of inputs differ at dimension {i}: "
                    f"{a_shape[i]} versus {b_shape[i]}."
                )
    return B.reshape(B.multiply(a[a_indices], b[b_indices]), *target_shape)


@dispatch
@abstract()
def svd(a: Numeric, compute_uv: bool = True):  # pragma: no cover
    """Compute the singular value decomposition.

    Args:
        a (tensor): Matrix to decompose.
        compute_uv (bool, optional): Also compute `U` and `V`. Defaults to
            `True`.

    Returns:
        tuple: `(U, S, V)` if `compute_uv` is `True` and just `S` otherwise.
    """


@dispatch
@abstract()
def eig(a: Numeric, compute_eigvecs: bool = True):  # pragma: no cover
    """Compute the eigenvalue decomposition.

    Args:
        a (tensor): Matrix to decompose.
        compute_eigvecs (bool, optional): Also compute eigenvectors. Defaults to `True`.

    Returns:
        tuple: `(S, V)` if `compute_eigvecs` is `True` and just `S` otherwise.
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
def pinv(a):
    """Compute the pseudo-inverse of `a`.

    Args:
        a (tensor): Matrix to compute pseudo-inverse of.

    Returns:
        tensor: Pseudo-inverse of `a`.
    """
    if B.shape(a, -2) >= B.shape(a, -1):
        chol = B.chol(B.matmul(a, a, tr_a=True))
        return B.cholsolve(chol, B.transpose(a))
    else:
        chol = B.chol(B.matmul(a, a, tr_b=True))
        return B.transpose(B.cholsolve(chol, a))


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


cholesky_retry_factor = 1
"""float: If the Cholesky decomposition throws an exception, increase `B.epsilon` by
this at most factor and try the Cholesky decomposition again."""


@dispatch
def cholesky(a: Numeric):
    """Compute the Cholesky decomposition. The matrix will automatically be regularised
    because computing the decomposition.

    Args:
        a (tensor): Matrix to decompose.

    Returns:
        tensor: Cholesky decomposition.
    """
    factor = 1
    while True:
        try:
            return _cholesky(reg(a, diag=factor * B.epsilon))
        except Exception as e:
            if factor < B.cholesky_retry_factor:
                # If we can still increase the factor, increase it and retry the
                # Cholesky.
                factor *= 10
                warnings.warn(
                    f"Cholesky decomposition failed. "
                    f"Trying again with regularisation `{factor * B.epsilon}`.",
                    stacklevel=2,
                )
                continue
            else:
                # We have increased the factor as much as we're allowed to. Throw
                # the original exception.
                raise e


chol = cholesky  #: Shorthand for `cholesky`.


@dispatch
@abstract()
def _cholesky(a: Numeric):  # pragma: no cover
    pass


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
def triangular_solve(a, b, lower_a: bool = True):  # pragma: no cover
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


def _a_b_uprank(a, b):
    a = B.uprank(a)
    b = B.uprank(b)
    return a, b


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
    if B.shape(a, -1) == 1 and B.shape(b, -1) == 1:
        return a * B.transpose(b)

    return B.matmul(a, b, tr_b=True)


@dispatch
def outer(a):
    return outer(a, a)


@dispatch
def reg(a, diag=None, clip: bool = True):
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
    if B.shape(a, -1) == 1 and B.shape(b, -1) == 1:
        return (a - B.transpose(b)) ** 2

    norms_a = B.sum(a**2, axis=-1)[..., :, None]
    norms_b = B.sum(b**2, axis=-1)[..., None, :]
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
    if B.shape(a, -1) == 1 and B.shape(b, -1) == 1:
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
    if B.shape(a, -1) == 1 and B.shape(b, -1) == 1:
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
    if B.shape(a, -1) == 1 and B.shape(b, -1) == 1:
        return (a + B.transpose(b)) ** 2

    norms_a = B.sum(a**2, axis=-1)[..., :, None]
    norms_b = B.sum(b**2, axis=-1)[..., None, :]
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
    if B.shape(a, -1) == 1 and B.shape(b, -1) == 1:
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
    if B.shape(a, -1) == 1 and B.shape(b, -1) == 1:
        return B.abs(a + b)

    return B.sqrt(B.maximum(B.ew_sums2(a, b), B.cast(B.dtype(a), 1e-30)))


@dispatch
def ew_sums(a):
    return ew_sums(a, a)
