# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from . import dispatch, Numeric


@dispatch(Numeric, Numeric)
def matmul(a, b, tr_a=False, tr_b=False):
    a = a.T if tr_a else a
    b = b.T if tr_b else b
    return np.matmul(a, b)
