# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from . import dispatch
from ..types import NPNumeric, NPDType, NPShape

__all__ = []


@dispatch(NPShape, NPDType)
def zeros(shape, dtype):
    return np.zeros(shape, dtype=dtype)


@dispatch(NPShape, NPDType)
def ones(shape, dtype):
    return np.ones(shape, dtype=dtype)


@dispatch(NPNumeric, NPDType)
def cast(a, dtype):
    return a.astype(dtype)


@dispatch(NPNumeric)
def abs(a):
    return np.abs(a)
