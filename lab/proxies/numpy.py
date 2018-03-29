# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np_
import autograd.numpy as np
import scipy.linalg as sla
# noinspection PyUnresolvedReferences
from autograd.numpy import *

__all__ = np_.__all__ + ['trisolve', 'randn', 'cholesky', 'concat']


def matmul(a, b, tr_a=False, tr_b=False):
    """Multiply two matrices.

    Args:
        a (matrix): Left hand side of the product.
        b (matrix): Right hand side of the product.
        tr_a (bool, optional): Transpose `a`. Defaults to `False`.
        tr_b (bool, optional): Transpose `b`. Defaults to `False`.
    """
    a = a.T if tr_a else a
    b = b.T if tr_b else b
    return np.matmul(a, b)


def trisolve(a, b, tr_a=False, lower=True):
    """Compute `inv(a) b` where `a` is triangular.

    Args:
        a (matrix): `a`.
        b (matrix): `b`.
        tr_a (bool, optional): Transpose `a` before inverting.
        lower (bool, optional): Indicate that `a` is lower triangular.
            Defaults to `True`.
    """
    return sla.solve_triangular(a, b,
                                trans='T' if tr_a else 'N',
                                lower=lower)


def randn(shape, dtype=None):
    """Generate standard random normal numbers.

    Args:
        shape (shape): Shape of output.
        dtype (data type, optional): Data type of output.
    """
    return cast(np.random.randn(*shape), dtype=dtype)


def cast(a, dtype=None):
    """Cast an object to a data type.

    Args:
        a (tensor): Object to cast.
        dtype (data type, optional): Data type to cast to.

    """
    return np.array(a) if dtype is None else np.array(a).astype(dtype)


cholesky = np.linalg.cholesky  #: Compute the Cholesky decomposition.
eig = np.linalg.eig  #: Compute the eigendecomposition.
dot = matmul  #: Multiply two matrices.
concat = np.concatenate  #: Concatenate tensors.
