# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from . import dispatch, Numeric

__all__ = []


@dispatch(Numeric, Numeric)
def matmul(a, b, tr_a=False, tr_b=False):
    a = a.T if tr_a else a
    b = b.T if tr_b else b
    return np.matmul(a, b)


@dispatch(Numeric)
def transpose(a):
    return a.T


@dispatch(Numeric)
def trace(a, axis1=0, axis2=1):
    return np.trace(a, axis1=axis1, axis2=axis2)


@dispatch(Numeric, Numeric)
def kron(a, b):
    return np.kron(a, b)


@dispatch(Numeric)
def svd(a, compute_uv=True):
    res = np.linalg.svd(a, full_matrices=True, compute_uv=compute_uv)
    return (res[0], res[1], res[2].T.conj()) if compute_uv else res
