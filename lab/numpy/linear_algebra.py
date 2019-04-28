# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.linalg as sla

from . import dispatch, NP

__all__ = []


@dispatch(NP, NP)
def matmul(a, b, tr_a=False, tr_b=False):
    a = a.T if tr_a else a
    b = b.T if tr_b else b
    return np.matmul(a, b)


@dispatch(NP)
def transpose(a):
    return a.T


@dispatch(NP)
def trace(a, axis1=0, axis2=1):
    return np.trace(a, axis1=axis1, axis2=axis2)


@dispatch(NP, NP)
def kron(a, b):
    return np.kron(a, b)


@dispatch(NP)
def svd(a, compute_uv=True):
    res = np.linalg.svd(a, full_matrices=True, compute_uv=compute_uv)
    return (res[0], res[1], res[2].T.conj()) if compute_uv else res


@dispatch(NP)
def cholesky(a):
    return np.linalg.cholesky(a)


@dispatch(NP, NP)
def cholesky_solve(a, b):
    return trisolve(a.T, trisolve(a, b), lower_a=False)


@dispatch(NP, NP)
def trisolve(a, b, lower_a=True):
    return sla.solve_triangular(a, b, trans='N', lower=lower_a)
