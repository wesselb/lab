from typing import Union, Optional

import tensorflow as tf

from . import dispatch, B, Numeric
from .custom import tensorflow_register
from ..custom import toeplitz_solve, s_toeplitz_solve, expm, s_expm, logm, s_logm
from ..linear_algebra import _default_perm
from ..types import Int
from ..util import resolve_axis

__all__ = []


@dispatch
def matmul(a: Numeric, b: Numeric, tr_a: bool = False, tr_b: bool = False):
    return tf.matmul(a, b, transpose_a=tr_a, transpose_b=tr_b)


@dispatch
def transpose(a: Numeric, perm: Optional[Union[tuple, list]] = None):
    # Correctly handle special cases.
    rank_a = B.rank(a)
    if rank_a == 0:
        return a
    elif rank_a == 1 and perm is None:
        return a[None, :]

    if perm is None:
        perm = _default_perm(a)
    return tf.transpose(a, perm=perm)


@dispatch
def trace(a: Numeric, axis1: Int = -2, axis2: Int = -1):
    axis1 = resolve_axis(a, axis1)
    axis2 = resolve_axis(a, axis2)
    perm = [i for i in range(B.rank(a)) if i not in [axis1, axis2]]
    perm += [axis1, axis2]
    a = tf.transpose(a, perm=perm)
    return tf.linalg.trace(a)


@dispatch
def svd(a: Numeric, compute_uv: bool = True):
    res = tf.linalg.svd(a, full_matrices=False, compute_uv=compute_uv)
    return (res[1], res[0], res[2]) if compute_uv else res


@dispatch
def eig(a: Numeric, compute_eigvecs: bool = True):
    vals, vecs = tf.linalg.eig(a)
    return (vals, vecs) if compute_eigvecs else vals


@dispatch
def solve(a: Numeric, b: Numeric):
    return tf.linalg.solve(a, b)


@dispatch
def inv(a: Numeric):
    return tf.linalg.inv(a)


@dispatch
def det(a: Numeric):
    return tf.linalg.det(a)


@dispatch
def logdet(a: Numeric):
    return tf.linalg.logdet(a)


_expm = tensorflow_register(expm, s_expm)


@dispatch
def expm(a: Numeric):
    return _expm(a)


_logm = tensorflow_register(logm, s_logm)


@dispatch
def logm(a: Numeric):
    return _logm(a)


@dispatch
def _cholesky(a: Numeric):
    return tf.linalg.cholesky(a)


@dispatch
def cholesky_solve(a: Numeric, b: Numeric):
    return tf.linalg.cholesky_solve(a, b)


@dispatch
def triangular_solve(a: Numeric, b: Numeric, lower_a: bool = True):
    return tf.linalg.triangular_solve(a, b, lower=lower_a)


_toeplitz_solve = tensorflow_register(toeplitz_solve, s_toeplitz_solve)


@dispatch
def toeplitz_solve(a: Numeric, b: Numeric, c: Numeric):
    return _toeplitz_solve(a, b, c)
