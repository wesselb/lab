# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from . import dispatch, B
from .types import Numeric, DType, Shape, default_dtype
from .util import abstract

__all__ = ['zeros',
           'ones',
           'cast',
           'abs']


@dispatch(Shape, DType)
@abstract(promote_to=None)
def zeros(shape, dtype):  # pragma: no cover
    """Create a tensor of zeros.

    Args:
        shape (shape or tensor): Shape of the tensor.
        dtype (dtype, optional): Data type. Defaults to `.types.default_dtype`
            or the data type of the provided tensor.
    """


@dispatch(Shape)
def zeros(shape):
    return zeros(shape, default_dtype)


@dispatch(Shape, Numeric)
def zeros(shape, ref):
    return zeros(shape, B.dtype(ref))


@dispatch(Numeric, DType)
def zeros(ref, dtype):
    return zeros(B.shape(ref), dtype)


@dispatch(Numeric)
def zeros(ref):
    return zeros(B.shape(ref), B.dtype(ref))


@dispatch(Shape, DType)
@abstract(promote_to=None)
def ones(shape, dtype):  # pragma: no cover
    """Create a tensor of ones.

    Args:
        shape (shape or tensor): Shape of the tensor.
        dtype (dtype, optional): Data type. Defaults to `.types.default_dtype`
            or the data type of the provided tensor.
    """


@dispatch(Shape)
def ones(shape):
    return ones(shape, default_dtype)


@dispatch(Shape, Numeric)
def ones(shape, ref):
    return ones(shape, B.dtype(ref))


@dispatch(Numeric, DType)
def ones(ref, dtype):
    return ones(B.shape(ref), dtype)


@dispatch(Numeric)
def ones(ref):
    return ones(B.shape(ref), B.dtype(ref))


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
