import logging

import numpy as np
import scipy.linalg as sla

from . import dispatch, B, Numeric
from ..custom import toeplitz_solve as _toeplitz_solve, expm as _expm, logm as _logm
from ..linear_algebra import _default_perm
from ..util import batch_computation

__all__ = []

log = logging.getLogger(__name__)


@dispatch
def matmul(a: Numeric, b: Numeric, tr_a=False, tr_b=False):
    a = transpose(a) if tr_a else a
    b = transpose(b) if tr_b else b
    return np.matmul(a, b)


@dispatch
def transpose(a: Numeric, perm=None):
    # Correctly handle special cases.
    rank_a = B.rank(a)
    if rank_a == 0:
        return a
    elif rank_a == 1 and perm is None:
        return a[None, :]

    if perm is None:
        perm = _default_perm(a)
    return np.transpose(a, axes=perm)


@dispatch
def trace(a: Numeric, axis1=-2, axis2=-1):
    return np.trace(a, axis1=axis1, axis2=axis2)


@dispatch
def kron(a: Numeric, b: Numeric):
    return np.kron(a, b)


@dispatch
def svd(a: Numeric, compute_uv=True):
    res = np.linalg.svd(a, full_matrices=False, compute_uv=compute_uv)
    return (res[0], res[1], np.conj(transpose(res[2]))) if compute_uv else res


@dispatch
def solve(a: Numeric, b: Numeric):
    return np.linalg.solve(a, b)


@dispatch
def inv(a: Numeric):
    return np.linalg.inv(a)


@dispatch
def det(a: Numeric):
    return np.linalg.det(a)


@dispatch
def logdet(a: Numeric):
    return np.linalg.slogdet(a)[1]


@dispatch
def expm(a: Numeric):
    return _expm(a)


@dispatch
def logm(a: Numeric):
    return _logm(a)


@dispatch
def cholesky(a: Numeric):
    return np.linalg.cholesky(a)


@dispatch
def cholesky_solve(a: Numeric, b: Numeric):
    return triangular_solve(transpose(a), triangular_solve(a, b), lower_a=False)


@dispatch
def triangular_solve(a: Numeric, b: Numeric, lower_a=True):
    def _triangular_solve(a_, b_):
        return sla.solve_triangular(
            a_, b_, trans="N", lower=lower_a, check_finite=False
        )

    return batch_computation(_triangular_solve, (a, b), (2, 2))


@dispatch
def toeplitz_solve(a: Numeric, b: Numeric, c: Numeric):
    return _toeplitz_solve(a, b, c)
