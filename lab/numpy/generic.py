# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from . import dispatch
from ..types import NPNumeric

__all__ = []


@dispatch(NPNumeric)
def abs(a):
    return np.abs(a)
