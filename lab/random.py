# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import torch

from . import dispatch
from .types import Dimension, DType, default_dtype, Int
from .util import abstract

__all__ = ['set_random_seed', 'rand', 'randn']


@dispatch(Int)
def set_random_seed(seed):
    """Set the random seed for all frameworks.

    Args:
        seed (int): Seed.
    """
    np.random.seed(seed)
    tf.set_random_seed(seed)
    torch.manual_seed(seed)


@dispatch(DType, [Dimension])
@abstract()
def rand(dtype, *shape):  # pragma: no cover
    """Construct a U[0, 1] random tensor.

    Args:
        dtype (dtype, optional): Data type. Defaults to `.types.default_dtype`.
        *shape (shape, optional): Shape of the tensor. Defaults to `()`.

    Returns:
        tensor: Random tensor.
    """


@dispatch([Dimension])
def rand(*shape):
    return rand(default_dtype, *shape)


@dispatch(DType, [Dimension])
@abstract(promote=None)
def randn(dtype, *shape):  # pragma: no cover
    """Construct a N(0, 1) random tensor.

    Args:
        dtype (dtype, optional): Data type. Defaults to `.types.default_dtype`.
        *shape (shape, optional): Shape of the tensor. Defaults to `()`.

    Returns:
        tensor: Random tensor.
    """


@dispatch([Dimension])
def randn(*shape):
    return randn(default_dtype, *shape)
