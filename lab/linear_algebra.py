# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from plum import promote

from . import dispatch, B
from .types import Numeric
from .util import abstract

__all__ = ['epsilon',
           'transpose', 't', 'T',
           'matmul', 'mm', 'dot',
           'kron',
           'trace',
           'svd',
           'cholesky',
           'cholesky_solve',
           'trisolve',
           'outer',
           'reg',
           'pw_dists2', 'pw_dists', 'ew_dists2', 'ew_dists',
           'pw_sums2', 'pw_sums', 'ew_sums2', 'ew_sums']

epsilon = 1e-12  #: Magnitude of diagonal to regularise matrices with.


def _default_perm(a):
    return tuple(range(B.rank(a) - 1, -1, -1))


@dispatch(Numeric)
@abstract()
def transpose(a, perm=None):  # pragma: no cover
    """Transpose a matrix.

    Args:
        a (tensor): Matrix to transpose.
        perm (list[int] or tuple[int], optional): Permutation. Defaults to
            reversing the ordering of all axes.

    Returns:
        tensor: Transposition of `a`.
    """


t = transpose  #: Shorthand for `transpose`.
T = transpose  #: Shorthand for `transpose`.


@dispatch(object, object)
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
dot = matmul  #: Also a shorthand for `matmul`.


@dispatch(Numeric)
@abstract()
def trace(a, axis1=0, axis2=1):  # pragma: no cover
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


@dispatch(object, object)
@abstract(promote=2)
def kron(a, b):  # pragma: no cover
    """Kronecker product.

    Args:
        a (tensor): First matrix.
        b (tensor): Second matrix.

    Returns:
        tensor: Kronecker product of `a` and `b`.
    """


@dispatch(Numeric)
@abstract()
def svd(a, compute_uv=True):  # pragma: no cover
    """Compute the singular value decomposition.

    Args:
        a (tensor): Matrix to decompose.
        compute_uv (bool, optional): Also compute `U` and `V`. Defaults to
            `True`.

    Returns:
        tuple: `(U, S, V)` is `compute_uv` is `True` and just `S` otherwise.
    """


@dispatch(Numeric)
@abstract()
def cholesky(a):  # pragma: no cover
    """Compute the Cholesky decomposition.

    Args:
        a (tensor): Matrix to decompose.

    Returns:
        tensor: Cholesky decomposition.
    """


@dispatch(object, object)
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


@dispatch(object, object)
@abstract(promote=2)
def trisolve(a, b, lower_a=True):  # pragma: no cover
    """Solve the linear system `a x = b` where `a` is triangular.

    Args:
        a (tensor): Triangular matrix `a`.
        b (tensor): RHS `b`.
        lower_a (bool, optional): Indicate that `a` is lower triangular
            instead of upper triangular. Defaults to `True`.

    Returns:
        tensor: Solution `x`.
    """


@dispatch(object, object)
def outer(a, b):
    """Compute the outer product between two vectors.

    Args:
        a (vector): First vector.
        b (vector): Second vector.

    Returns:
        tensor: Outer product of `a` and `b`.
    """
    if B.rank(a) != 1 or B.rank(b) != 1:
        raise ValueError('Arguments must have rank 1.')
    return B.expand_dims(a, axis=1) * B.expand_dims(b, axis=0)


@dispatch(object)
def outer(a):
    return outer(a, a)


@dispatch(object)
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
    if clip:
        diag = B.maximum(diag, B.epsilon)
    return a + diag * B.eye(a)


@dispatch(object, object)
def pw_dists2(a, b):
    """Compute the square the Euclidean norm of the pairwise
    differences between two matrices where rows correspond to elements and
    columns to features.

    Args:
        a (matrix): First matrix.
        b (matrix, optional): Second matrix. Defaults to `a`.

    Returns:
        matrix: Square of the Euclidean norm of the pairwise differences
            between the elements of `a` and `b`.
    """
    # Optimise the one-dimensional case.
    if B.shape(a)[1] == 1 and B.shape(b)[1] == 1:
        return (a - B.transpose(b)) ** 2
    norms_a = B.sum(a ** 2, axis=1)[:, None]
    norms_b = B.sum(b ** 2, axis=1)[None, :]
    return norms_a + norms_b - 2 * B.matmul(a, b, tr_b=True)


@dispatch(object)
def pw_dists2(a):
    return pw_dists2(a, a)


@dispatch(object, object)
def pw_dists(a, b):
    """Compute the Euclidean norm of the pairwise differences between two
    matrices where rows correspond to elements and columns to features.

    Args:
        a (matrix): First matrix.
        b (matrix, optional): Second matrix. Defaults to `a`.

    Returns:
        matrix: Euclidean norm of the pairwise differences between the
            elements of `a` and `b`.
    """
    # Optimise the one-dimensional case.
    if B.shape(a)[1] == 1 and B.shape(b)[1] == 1:
        return B.abs(a - B.transpose(b))
    return B.sqrt(B.maximum(B.pw_dists2(a, b),
                            B.cast(1e-30, B.dtype(a))))


@dispatch(object)
def pw_dists(a):
    return pw_dists(a, a)


@dispatch(object, object)
def ew_dists2(a, b):
    """Compute the square the Euclidean norm of the element-wise
    differences between two matrices where rows correspond to elements and
    columns to features.

    Args:
        a (matrix): First matrix.
        b (matrix, optional): Second matrix. Defaults to `a`.

    Returns:
        matrix: Square of the Euclidean norm of the element-wise differences
            between the elements of `a` and `b`.
    """
    return B.sum((a - b) ** 2, axis=1)[:, None]


@dispatch(object)
def ew_dists2(a):
    return ew_dists2(a, a)


@dispatch(object, object)
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
    # Optimise the one-dimensional case.
    if B.shape(a)[1] == 1 and B.shape(b)[1] == 1:
        return B.abs(a - b)
    return B.sqrt(B.maximum(B.ew_dists2(a, b),
                            B.cast(1e-30, B.dtype(a))))


@dispatch(object)
def ew_dists(a):
    return ew_dists(a, a)


@dispatch(object, object)
def pw_sums2(a, b):
    """Compute the square the Euclidean norm of the pairwise
    sums between two matrices where rows correspond to elements and
    columns to features.

    Args:
        a (matrix): First matrix.
        b (matrix, optional): Second matrix. Defaults to `a`.

    Returns:
        matrix: Square of the Euclidean norm of the pairwise sums
            between the elements of `a` and `b`.
    """
    # Optimise the one-dimensional case.
    if B.shape(a)[1] == 1 and B.shape(b)[1] == 1:
        return (a + B.transpose(b)) ** 2
    norms_a = B.sum(a ** 2, axis=1)[:, None]
    norms_b = B.sum(b ** 2, axis=1)[None, :]
    return norms_a + norms_b + 2 * B.matmul(a, b, tr_b=True)


@dispatch(object)
def pw_sums2(a):
    return pw_sums2(a, a)


@dispatch(object, object)
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
    # Optimise the one-dimensional case.
    if B.shape(a)[1] == 1 and B.shape(b)[1] == 1:
        return B.abs(a + B.transpose(b))
    return B.sqrt(B.maximum(B.pw_sums2(a, b),
                            B.cast(1e-30, B.dtype(a))))


@dispatch(object)
def pw_sums(a):
    return pw_sums(a, a)


@dispatch(object, object)
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
    return B.sum((a + b) ** 2, axis=1)[:, None]


@dispatch(object)
def ew_sums2(a):
    return ew_sums2(a, a)


@dispatch(object, object)
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
    # Optimise the one-dimensional case.
    if B.shape(a)[1] == 1 and B.shape(b)[1] == 1:
        return B.abs(a + b)
    return B.sqrt(B.maximum(B.ew_sums2(a, b),
                            B.cast(1e-30, B.dtype(a))))


@dispatch(object)
def ew_sums(a):
    return ew_sums(a, a)
