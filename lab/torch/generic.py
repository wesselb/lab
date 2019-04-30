# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import torch

from . import dispatch
from ..types import TorchNumeric, TorchDType

__all__ = []


@dispatch(TorchNumeric, TorchDType)
def cast(a, dtype):
    return a.type(dtype)


@dispatch(TorchNumeric)
def abs(a):
    return torch.abs(a)
