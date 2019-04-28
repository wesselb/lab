# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from . import dispatch, NP

__all__ = []


@dispatch(NP)
def abs(a):
    return np.abs(a)
