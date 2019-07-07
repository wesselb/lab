# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

import autograd.numpy as anp
import autograd.scipy.linalg as asla

from . import dispatch, B
from .custom import autograd_register
from ..custom import toeplitz_solve, s_toeplitz_solve
from ..linear_algebra import _default_perm
from ..types import NPNumeric
from ..util import batch_computation

__all__ = []
log = logging.getLogger(__name__)


@dispatch(NPNumeric, NPNumeric)
def matmul(a, b, tr_a=False, tr_b=False):
    a = transpose(a) if tr_a else a
    b = transpose(b) if tr_b else b
    return anp.matmul(a, b)


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
    return anp.transpose(a, axes=perm)


@dispatch(NPNumeric)
def trace(a, axis1=0, axis2=1):
    if axis1 == axis2:
        raise ValueError('Keyword argument axis1 and axis2 cannot be the same.')

    # AutoGrad does not support the `axis1` and `axis2` arguments...

    # Order the axis as `axis1 < axis`.
    if axis2 < axis1:
        axis1, axis2 = axis2, axis1

    # Bring the trace axes forward.
    if (axis1, axis2) != (0, 1):
        perm = [axis1, axis2] + \
               [i for i in range(B.rank(a)) if i != axis1 and i != axis2]
        a = anp.transpose(a, axes=perm)

    return anp.trace(a)


@dispatch(NPNumeric, NPNumeric)
def kron(a, b):
    return anp.kron(a, b)


@dispatch(NPNumeric)
def svd(a, compute_uv=True):
    res = anp.linalg.svd(a, full_matrices=False, compute_uv=compute_uv)
    return (res[0], res[1], transpose(res[2]).conj()) if compute_uv else res


@dispatch(NPNumeric, NPNumeric)
def solve(a, b):
    return anp.linalg.solve(a, b)


@dispatch(NPNumeric)
def inv(a):
    return anp.linalg.inv(a)


@dispatch(NPNumeric)
def det(a):
    return anp.linalg.det(a)


@dispatch(NPNumeric)
def logdet(a):
    return anp.linalg.slogdet(a)[1]


@dispatch(NPNumeric)
def cholesky(a):
    return anp.linalg.cholesky(a)


@dispatch(NPNumeric, NPNumeric)
def cholesky_solve(a, b):
    return triangular_solve(transpose(a), triangular_solve(a, b), lower_a=False)


@dispatch(NPNumeric, NPNumeric)
def triangular_solve(a, b, lower_a=True):
    def _triangular_solve(a_, b_):
        return asla.solve_triangular(a_, b_,
                                     trans='N',
                                     lower=lower_a,
                                     check_finite=False)

    return batch_computation(_triangular_solve, (a, b), (2, 2))


f = autograd_register(toeplitz_solve, s_toeplitz_solve)
dispatch(NPNumeric, NPNumeric, NPNumeric)(f)
