# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from . import dispatch
from .types import Numeric
from .util import abstract

__all__ = ['shape',
           'shape_int',
           'rank',
           'length']


@dispatch(Numeric)
@abstract
def shape(a):  # pragma: no cover
    """Get the shape of a tensor.

    Args:
        a (tensor): Tensor.

    Returns:
        object: Shape of `a`.
    """
    pass


@dispatch(Numeric)
@abstract
def shape_int(a):  # pragma: no cover
    """Get the shape of a tensor as a tuple of integers.

    Args:
        a (tensor): Tensor.

    Returns:
        tuple: Shape of `a` as a tuple of integers.
    """
    pass


@dispatch(Numeric)
@abstract
def rank(a):  # pragma: no cover
    """Get the shape of a tensor.

    Args:
        a (tensor): Tensor.

    Returns:
        object: Shape of `a`.
    """
    pass


@dispatch(Numeric)
@abstract
def length(a):  # pragma: no cover
    """Get the length of a tensor.

    Args:
        a (tensor): Tensor.

    Returns:
        object: Shape of `a`.
    """
    pass
