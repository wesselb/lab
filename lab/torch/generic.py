# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import torch

from . import dispatch, Numeric

__all__ = []


@dispatch(Numeric)
def abs(a):
    return torch.abs(a)
