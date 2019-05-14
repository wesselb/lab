# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import autograd.numpy as np
import autograd.scipy.linalg as sla

from . import dispatch
from ..linear_algebra import _default_perm
from ..types import NPNumeric

__all__ = []


@dispatch(NPNumeric, NPNumeric)
def matmul(a, b, tr_a=False, tr_b=False):
    a = a.T if tr_a else a
    b = b.T if tr_b else b
    return np.matmul(a, b)


@dispatch(NPNumeric)
def transpose(a, perm=None):
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
    res = np.linalg.svd(a, full_matrices=True, compute_uv=compute_uv)
    return (res[0], res[1], res[2].T.conj()) if compute_uv else res


@dispatch(NPNumeric, NPNumeric)
def solve(a, b):
    return np.linalg.solve(a, b)


@dispatch(NPNumeric)
def cholesky(a):
    return np.linalg.cholesky(a)


@dispatch(NPNumeric, NPNumeric)
def cholesky_solve(a, b):
    return trisolve(a.T, trisolve(a, b), lower_a=False)


@dispatch(NPNumeric, NPNumeric)
def trisolve(a, b, lower_a=True):
    return sla.solve_triangular(a, b, trans='N', lower=lower_a)
