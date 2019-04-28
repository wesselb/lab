# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import torch

from . import dispatch, Numeric


@dispatch(Numeric, Numeric)
def matmul(a, b, tr_a=False, tr_b=False):
    a = a.t() if tr_a else a
    b = b.t() if tr_b else b
    return torch.matmul(a, b)
