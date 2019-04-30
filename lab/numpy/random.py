# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from plum import convert

from . import dispatch
from ..types import NPShape, NPDType, NPNumeric

__all__ = []


@dispatch(NPShape, NPDType)
def rand(shape, dtype):
    return convert(np.random.rand(*shape), NPNumeric).astype(dtype)


@dispatch(NPShape, NPDType)
def randn(shape, dtype):
    return convert(np.random.randn(*shape), NPNumeric).astype(dtype)
