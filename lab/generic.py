# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from . import dispatch
from .types import Numeric
from .util import abstract

__all__ = ['abs']


@dispatch(Numeric)
@abstract
def abs(a):  # pragma: no cover
    """Absolute value.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Absolute value of `a`.
    """
    pass
