# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from . import dispatch, Numeric

__all__ = []


@dispatch(Numeric)
def shape(a):
    return np.shape(a)


@dispatch(Numeric)
def shape_int(a):
    return np.shape(a)


@dispatch(Numeric)
def rank(a):
    return a.ndim


@dispatch(Numeric)
def length(a):
    return np.size(a)
