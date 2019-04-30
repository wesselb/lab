# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import torch

from . import dispatch
from ..types import TorchNumeric, TorchDType, TorchShape

__all__ = []


@dispatch(TorchShape, TorchDType)
def zeros(shape, dtype):
    return torch.zeros(shape, dtype=dtype)


@dispatch(TorchShape, TorchDType)
def ones(shape, dtype):
    return torch.ones(shape, dtype=dtype)


@dispatch(TorchNumeric, TorchDType)
def cast(a, dtype):
    return a.type(dtype)


@dispatch(TorchNumeric)
def abs(a):
    return torch.abs(a)


@dispatch(TorchNumeric)
def exp(a):
    return torch.exp(a)


@dispatch(TorchNumeric)
def log(a):
    return torch.log(a)


@dispatch(TorchNumeric)
def sin(a):
    return torch.sin(a)


@dispatch(TorchNumeric)
def cos(a):
    return torch.cos(a)


@dispatch(TorchNumeric)
def tan(a):
    return torch.tan(a)


@dispatch(TorchNumeric)
def tanh(a):
    return torch.tanh(a)


@dispatch(TorchNumeric, TorchNumeric)
def add(a, b):
    return a + b


@dispatch(TorchNumeric, TorchNumeric)
def subtract(a, b):
    return a - b


@dispatch(TorchNumeric, TorchNumeric)
def multiply(a, b):
    return a * b


@dispatch(TorchNumeric, TorchNumeric)
def divide(a, b):
    return a / b


@dispatch(TorchNumeric, TorchNumeric)
def power(a, b):
    return torch.pow(a, b)


@dispatch(TorchNumeric, TorchNumeric)
def minimum(a, b):
    return torch.min(a, b)


@dispatch(TorchNumeric, TorchNumeric)
def maximum(a, b):
    return torch.max(a, b)
