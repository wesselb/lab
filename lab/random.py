# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import torch

from . import dispatch
from .types import Shape, DType, default_dtype
from .util import abstract

__all__ = ['set_random_seed', 'rand', 'randn']


def set_random_seed(seed):
    """Set the random seed for all frameworks.

    Args:
        seed (int): Seed.
    """
    np.random.seed(seed)
    tf.set_random_seed(seed)
    torch.manual_seed(seed)


@dispatch(Shape, DType)
@abstract(promote_to=None)
def rand(shape, dtype):  # pragma: no cover
    """Construct a U[0, 1] random tensor.

    Args:
        shape (shape, optional): Shape of the tensor. Defaults to `()`.
        dtype (dtype, optional): Data type. Defaults to `.dtype.default_dtype`.

    Returns:
        tensor: Random tensor.
    """


@dispatch(Shape)
def rand(shape):
    return rand(shape, default_dtype)


@dispatch(DType)
def rand(dtype):
    return rand((), dtype)


@dispatch()
def rand():
    return rand(())


@dispatch(Shape, DType)
@abstract(promote_to=None)
def randn(shape, dtype):  # pragma: no cover
    """Construct a N(0, 1) random tensor.

    Args:
        shape (shape, optional): Shape of the tensor. Defaults to `()`.
        dtype (dtype, optional): Data type. Defaults to `.dtype.default_dtype`.

    Returns:
        tensor: Random tensor.
    """


@dispatch(Shape)
def randn(shape):
    return randn(shape, default_dtype)


@dispatch(DType)
def randn(dtype):
    return randn((), dtype)


@dispatch()
def randn():
    return randn(())
