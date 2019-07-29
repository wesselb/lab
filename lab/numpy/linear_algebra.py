# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import scipy.linalg as sla

from . import dispatch, B
from ..custom import toeplitz_solve
from ..linear_algebra import _default_perm
from ..types import NPNumeric
from ..util import batch_computation

__all__ = []
log = logging.getLogger(__name__)


@dispatch(NPNumeric, NPNumeric)
def matmul(a, b, tr_a=False, tr_b=False):
    a = transpose(a) if tr_a else a
    b = transpose(b) if tr_b else b
    return np.matmul(a, b)


@dispatch(NPNumeric)
def transpose(a, perm=None):
    # Correctly handle special cases.
    rank_a = B.rank(a)
    if rank_a == 0:
        return a
    elif rank_a == 1 and perm is None:
        return a[None, :]

    if perm is None:
        perm = _default_perm(a)
    return np.transpose(a, axes=perm)


@dispatch(NPNumeric)
def trace(a, axis1=0, axis2=1):
    return np.trace(a, axis1=axis1, axis2=axis2)


@dispatch(NPNumeric, NPNumeric)
def kron(a, b):
    return np.kron(a, b)


@dispatch(NPNumeric)
def svd(a, compute_uv=True):
    res = np.linalg.svd(a, full_matrices=False, compute_uv=compute_uv)
    return (res[0], res[1], transpose(res[2]).conj()) if compute_uv else res


@dispatch(NPNumeric, NPNumeric)
def solve(a, b):
    return np.linalg.solve(a, b)


@dispatch(NPNumeric)
def inv(a):
    return np.linalg.inv(a)


@dispatch(NPNumeric)
def det(a):
    return np.linalg.det(a)


@dispatch(NPNumeric)
def logdet(a):
    return np.linalg.slogdet(a)[1]


@dispatch(NPNumeric)
def cholesky(a):
    return np.linalg.cholesky(a)


@dispatch(NPNumeric, NPNumeric)
def cholesky_solve(a, b):
    return triangular_solve(transpose(a), triangular_solve(a, b), lower_a=False)


@dispatch(NPNumeric, NPNumeric)
def triangular_solve(a, b, lower_a=True):
    def _triangular_solve(a_, b_):
        return sla.solve_triangular(a_, b_,
                                    trans='N',
                                    lower=lower_a,
                                    check_finite=False)

    return batch_computation(_triangular_solve, (a, b), (2, 2))


dispatch(NPNumeric, NPNumeric, NPNumeric)(toeplitz_solve)
