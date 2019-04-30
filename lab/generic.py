# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from . import dispatch
from .types import Numeric, DType
from .util import abstract

__all__ = ['cast',
           'abs']


@dispatch(Numeric, DType)
@abstract(promote_to=None)
def cast(a, dtype):  # pragma: no cover
    """Cast an object to another data type.

    Args:
        a (tensor): Tensor to cast.
        dtype (dtype): New data type.

    Returns:
        tensor: `a`, but of data type `dtype`.
    """


@dispatch(Numeric)
@abstract()
def abs(a):  # pragma: no cover
    """Absolute value.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Absolute value of `a`.
    """
