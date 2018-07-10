# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np
import torch
from plum import Dispatcher

from .. import B

_dispatch = Dispatcher()

_Numeric = {Number, torch.Tensor}
_DType = {type, np.dtype, torch.dtype}


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


def concat(a, axis):
    return torch.cat(a, axis)


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


def stack(a, axis):
    return torch.stack(a, dim=axis)


def unstack(a, axis):
    return torch.unbind(a, dim=axis)


def expand_dims(a, axis):
    return torch.tensor(a).unsqueeze(axis)


def Variable(a, dtype=None):
    dtype = B.dtype(a) if dtype is None else dtype
    return torch.tensor(a, dtype=dtype, requires_grad=True)


_dtype_map = {
    torch.float64: np.float64,
    torch.float32: np.float32,
    torch.int64: np.int64,
    torch.int32: np.int32
}


@_dispatch(type, type)
def issubdtype(a, b):
    return np.issubdtype(a, b)


@_dispatch(torch.dtype, type)
def issubdtype(a, b):
    return issubdtype(_dtype_map[a], b)


@_dispatch(type, torch.dtype)
def issubdtype(a, b):
    return issubdtype(a, _dtype_map[b])


@_dispatch(torch.dtype, torch.dtype)
def issubdtype(a, b):
    return issubdtype(_dtype_map[a], _dtype_map[b])


def take(a, indices, axis=0):
    if axis > 0:
        a = torch.transpose(a, 0, axis)
    a = a[indices]
    if axis > 0:
        a = torch.transpose(a, 0, axis)
    return a


def assign(a, value):
    a.data = torch.tensor(value).data


array = torch.tensor
dot = matmul

divide = torch.div
multiply = torch.mul
subtract = torch.sub

less = torch.lt
greater = torch.gt
less_equal = torch.le
greater_equal = torch.ge

maximum = torch.max
minimum = torch.min

# Neural net activations:
sigmoid = torch.nn.functional.sigmoid
tanh = torch.nn.functional.tanh
relu = torch.nn.functional.relu
leaky_relu = torch.nn.functional.leaky_relu
