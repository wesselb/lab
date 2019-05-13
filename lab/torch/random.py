# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import torch

from . import dispatch
from ..types import TorchDimension, TorchDType

__all__ = []


@dispatch(TorchDType, [TorchDimension])
def rand(dtype, *shape):
    return torch.rand(shape, dtype=dtype)


@dispatch(TorchDType, [TorchDimension])
def randn(dtype, *shape):
    return torch.randn(shape, dtype=dtype)
