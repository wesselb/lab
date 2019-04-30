# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from . import dispatch
from ..types import NPNumeric, NPDType

__all__ = []


@dispatch(NPNumeric, NPDType)
def cast(a, dtype):
    return a.astype(dtype)


@dispatch(NPNumeric)
def abs(a):
    return np.abs(a)
