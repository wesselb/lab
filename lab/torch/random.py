# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import torch

from . import dispatch
from ..types import TorchShape, TorchDType

__all__ = []


@dispatch(TorchShape, TorchDType)
def rand(shape, dtype):
    return torch.rand(shape, dtype=dtype)


@dispatch(TorchShape, TorchDType)
def randn(shape, dtype):
    return torch.randn(shape, dtype=dtype)
