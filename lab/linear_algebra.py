# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from plum import promote

from . import dispatch
from .types import Numeric


@dispatch(Numeric, Numeric)
def matmul(x, y, **kw_args):
    """Matrix multiplication.

    Args:
        x (tensor): First matrix.
        y (tensor): Second matrix.
        tr_x (bool, optional): Transpose first matrix. Defaults to `False`.
        tr_y (bool, optional): Transpose second matrix. Defaults to `False`.

    Returns:
        tensor: Matrix product of `x` and `y`.
    """
    return matmul(*promote(x, y), **kw_args)
