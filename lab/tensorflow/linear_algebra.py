import tensorflow as tf

from . import dispatch, B, Numeric
from .custom import tensorflow_register
from ..custom import toeplitz_solve, s_toeplitz_solve
from ..linear_algebra import _default_perm

__all__ = []


@dispatch(Numeric, Numeric)
def matmul(a, b, tr_a=False, tr_b=False):
    return tf.matmul(a, b, transpose_a=tr_a, transpose_b=tr_b)


@dispatch(Numeric)
def transpose(a, perm=None):
    # Correctly handle special cases.
    rank_a = B.rank(a)
    if rank_a == 0:
        return a
    elif rank_a == 1 and perm is None:
        return a[None, :]

    if perm is None:
        perm = _default_perm(a)
    return tf.transpose(a, perm=perm)


@dispatch(Numeric)
def trace(a, axis1=0, axis2=1):
    perm = [i for i in range(B.rank(a)) if i not in [axis1, axis2]]
    perm += [axis1, axis2]
    a = tf.transpose(a, perm=perm)
    return tf.linalg.trace(a)


@dispatch(Numeric, Numeric)
def kron(a, b):
    shape_a = B.shape(a)
    shape_b = B.shape(b)

    # Check that ranks are equal.
    if len(shape_a) != len(shape_b):
        raise ValueError('Inputs must have equal rank.')

    for i in range(len(shape_a)):
        a = tf.expand_dims(a, axis=2 * i + 1)
        b = tf.expand_dims(b, axis=2 * i)
    return tf.reshape(a * b, tuple(x * y for x, y in zip(shape_a, shape_b)))


@dispatch(Numeric)
def svd(a, compute_uv=True):
    res = tf.linalg.svd(a, full_matrices=False, compute_uv=compute_uv)
    return (res[1], res[0], res[2]) if compute_uv else res


@dispatch(Numeric, Numeric)
def solve(a, b):
    return tf.linalg.solve(a, b)


@dispatch(Numeric)
def inv(a):
    return tf.linalg.inv(a)


@dispatch(Numeric)
def det(a):
    return tf.linalg.det(a)


@dispatch(Numeric)
def logdet(a):
    return tf.linalg.logdet(a)


@dispatch(Numeric)
def cholesky(a):
    return tf.linalg.cholesky(a)


@dispatch(Numeric, Numeric)
def cholesky_solve(a, b):
    return tf.linalg.cholesky_solve(a, b)


@dispatch(Numeric, Numeric)
def triangular_solve(a, b, lower_a=True):
    return tf.linalg.triangular_solve(a, b, lower=lower_a)


f = tensorflow_register(toeplitz_solve, s_toeplitz_solve)
dispatch(Numeric, Numeric, Numeric)(f)
