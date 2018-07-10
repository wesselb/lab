# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import autograd.numpy as np
import autograd.scipy.linalg as sla
from plum import Dispatcher

from .. import B

_dispatch = Dispatcher()


def unstack(a, axis=0):
    """Inverse operation of stack.

    Args:
        a (tensor): Tensor to unstack.
        axis (int, optional): Axis along which to unstack. Defaults to `0`.
    """
    dims = len(np.shape(a))

    if axis > 0:
        # Switch `axis` and `0`.
        perm = list(range(dims))
        perm[axis] = 0
        perm[0] = axis
        a = np.transpose(a, perm)

    # Transpose and unstack
    size = np.shape(a)[0]
    slices = [a[i] for i in range(size)]

    if axis > 1:
        # Put the axis back in place.
        perm = list(range(dims - 1))
        perm.pop(axis - 1)
        perm = [axis - 1] + perm
        slices = [np.transpose(slice, perm) for slice in slices]

    # Return result.
    return slices


def matmul(a, b, tr_a=False, tr_b=False):
    """Multiply two matrices.

    Args:
        a (matrix): Left hand side of the product.
        b (matrix): Right hand side of the product.
        tr_a (bool, optional): Transpose `a`. Defaults to `False`.
        tr_b (bool, optional): Transpose `b`. Defaults to `False`.
    """
    if B.rank(a) == 0 and B.rank(b) == 0:
        return a * b
    a = B.transpose(a) if tr_a else a
    b = B.transpose(b) if tr_b else b
    return np.matmul(a, b)


def trisolve(a, b, tr_a=False, lower=True):
    """Compute `inv(a) b` where `a` is triangular.

    Args:
        a (matrix): `a`.
        b (matrix): `b`.
        tr_a (bool, optional): Transpose `a` before inverting.
        lower (bool, optional): Indicate that `a` is lower triangular.
            Defaults to `True`.
    """
    return sla.solve_triangular(a, b,
                                trans='T' if tr_a else 'N',
                                lower=lower)


def randn(shape, dtype=None):
    """Generate standard random normal numbers.

    Args:
        shape (shape): Shape of output.
        dtype (data type, optional): Data type of output.
    """
    return cast(np.random.randn(*shape), dtype=dtype)


def rand(shape, dtype=None):
    """Generate standard uniform normal numbers.

    Args:
        shape (shape): Shape of output.
        dtype (data type, optional): Data type of output.
    """
    return cast(np.random.rand(*shape), dtype=dtype)


def cast(a, dtype=None):
    """Cast an object to a data type.

    Args:
        a (tensor): Object to cast.
        dtype (data type, optional): Data type to cast to.

    """
    return a if dtype is None else np.array(a).astype(dtype)


def logsumexp(a, dtype=None):
    """Exponentiate, sum, and compute the logarithm of the result.

    Args:
        a (tensor): Array to perform computation on.
        dtype (data type, optional): Data type to cast to.
    """
    m = np.max(a)
    return cast(np.log(np.sum(np.exp(a - m))) + m, dtype=dtype)


def cholesky_solve(chol, a):
    """Solve a system given a Cholesky factorisation.

    Args:
        chol (tensor): Cholesky factorisation.
        a (tensor): Coefficients.
    """
    return trisolve(chol, trisolve(chol, a), tr_a=True)


def take(a, indices, axis=0):
    """Take particular indices from an axis.

    Args:
        a (tensor): Tensor to pick apart.
        indices (tuple or list): Indices to pick.
        axis (int, optional): Axis to pick from.

    Returns:
        tensor: `a` after taking `indices` from `axis`.
    """
    return np.take(a, indices, axis)


def svd(a, full_matrices=False, compute_uv=True):
    """Compute the singular value decomposition.

    Args:
        a (tensor): Matrix to compute SVD of.
        full_matrices (bool, optional): Compute a full or truncated SDV.
            Default to `False`.
        compute_uv (bool, optional): Also compute `U` and `V`. Defaults to
            `True`.

    Returns:
        tuple: `(U, S, V)` is `compute_uv` is `True` and just `S` otherwise.
    """
    res = np.linalg.svd(a,
                        full_matrices=full_matrices,
                        compute_uv=compute_uv)
    return res[0], res[1], res[2].T.conj() if compute_uv else res


def sigmoid(x):
    """Sigmoid function."""
    return 1. / (1. + np.exp(x))


def relu(x):
    """ReLU."""
    return np.maximum(x, 0, x)


def leaky_relu(x, alpha=0.2):
    """ReLU.

    Args:
        x (tensor): Input.
        alpha (tensor): Leak.

    Returns:
        tensor: Activation.
    """
    return np.maximum(x, alpha * x, x)


def vec_to_tril(a):
    """Convert a vector to a lower-triangular matrix.

    Args:
        a (tensor): Vector.

    Returns:
        tensor: Lower triangular matrix filled with `a`.
    """
    if B.rank(a) != 1:
        raise ValueError('Input must be rank 1.')

    # Figure out shape of output.
    n = B.shape_int(a)[0]
    m = int(((1 + 8 * n) ** .5 - 1) / 2)

    # Construct output and return.
    out = np.zeros((m, m))
    out[np.tril_indices(m)] = a
    return out


def tril_to_vec(a):
    """Convert a lower-triangular matrix to a vector.

    Args:
        a (tensor): Lower-triangular matrix.

    Returns:
        tensor: The lower-triangular part of `a` as a vector.
    """
    if B.rank(a) != 2:
        raise ValueError('Input must be rank 2.')
    n, m = B.shape_int(a)
    if n != m:
        raise ValueError('input must be square')
    return a[np.tril_indices(n)]


cholesky = np.linalg.cholesky  #: Compute the Cholesky decomposition.
eig = np.linalg.eig  #: Compute the eigendecomposition.
dot = matmul  #: Multiply two matrices.
concat = np.concatenate  #: Concatenate tensors.



@_dispatch(object)
def diag(a): return np.diag(a)
