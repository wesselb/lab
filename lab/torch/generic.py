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
