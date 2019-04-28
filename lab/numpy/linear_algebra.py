# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from . import dispatch, Numeric


@dispatch(Numeric, Numeric)
def matmul(a, b):
    return np.matmul(a, b)