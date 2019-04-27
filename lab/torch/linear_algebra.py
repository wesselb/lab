# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import torch

from . import dispatch, Numeric


@dispatch(Numeric, Numeric)
def matmul(x, y):
    return torch.matmul(x, y)
