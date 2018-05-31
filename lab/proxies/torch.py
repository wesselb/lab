# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import torch

_Numeric = {int, float, np.ndarray, torch.Tensor}


def shape(a):
    return a.shape


def rank(a):
    return len(a.shape)


def transpose(a):
    return torch.transpose(a, 0, 1)


def matmul(a, b, tr_a=False, tr_b=False):
    a = transpose(a) if tr_a else a
    b = transpose(b) if tr_b else b
    return torch.mm(a, b)


def sum(a, axis=None):
    if axis is None:
        return torch.sum(a)
    else:
        return torch.sum(a, dim=axis)


def cast(a, dtype=None):
    return a if dtype is None else torch.tensor(a).type(dtype)


def cholesky(a, lower=True):
    return torch.potrf(a, upper=not lower)


def cholesky_solve(chol, a, lower=True):
    return torch.potrs(a, chol, upper=not lower)


def trisolve(a, b, tr_a=False, lower=True):
    return torch.trtrs(b, a, upper=not lower, transpose=tr_a)[0]


array = torch.tensor
dot = matmul
