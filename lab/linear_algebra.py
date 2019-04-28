# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from plum import promote

from . import dispatch
from .util import abstract
from .types import Numeric

__all__ = ['matmul',
           'transpose']


@dispatch(Numeric, Numeric)
def matmul(a, b, **kw_args):
    """Matrix multiplication.

    Args:
        a (tensor): First matrix.
        b (tensor): Second matrix.
        tr_a (bool, optional): Transpose first matrix. Defaults to `False`.
        tr_b (bool, optional): Transpose second matrix. Defaults to `False`.

    Returns:
        tensor: Matrix product of `a` and `b`.
    """
    return matmul(*promote(a, b), **kw_args)


@dispatch(Numeric)
@abstract
def transpose(a):  # pragma: no cover
    """Transpose a matrix.

    Args:
        a (tensor): Matrix to transposed.

    Returns:
        tensor: Transposition of `a`.
    """
    pass
