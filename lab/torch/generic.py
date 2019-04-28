# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import torch

from . import dispatch, Torch

__all__ = []


@dispatch(Torch)
def abs(a):
    return torch.abs(a)
