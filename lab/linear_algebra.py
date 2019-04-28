# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from plum import promote

from . import dispatch
from .types import Numeric


@dispatch(Numeric, Numeric)
def matmul(a, b, **kw_args):
    """Matrix multiplication.

    Args:
        a (tensor): First matrix.
        b (tensor): Second matrix.
        tr_a (bool, optional): Transpose first matrix. Defaults to `False`.
        tr_b (bool, optional): Transpose second matrix. Defaults to `False`.

    Returns:
        tensor: Matrix product of `x` and `y`.
    """
    return matmul(*promote(a, b), **kw_args)
