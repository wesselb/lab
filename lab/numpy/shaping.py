# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from . import dispatch, NP

__all__ = []


@dispatch(NP)
def shape(a):
    return np.shape(a)


@dispatch(NP)
def shape_int(a):
    return np.shape(a)


@dispatch(NP)
def rank(a):
    return a.ndim


@dispatch(NP)
def length(a):
    return np.size(a)
