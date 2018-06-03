# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from plum import Dispatcher, PromisedType
from numbers import Number

from . import B

epsilon = 1e-12  #: Magnitude of diagonal to regularise matrices with.

_Numeric = {Number, np.ndarray}  #: Type of numerical objects.

_dispatch = Dispatcher()


def rank(a):
    """Get the rank of a tensor.

    Args:
        a (tensor): Tensor to get the rank of.
    """
    return B.shape(B.shape(a))[0]


def reg(a, diag=None, clip=True):
    """Add a diagonal to a matrix.

    Args:
        a (matrix): Matrix to add a diagonal to.
        diag (float, optional): Magnitude of the diagonal to add. Defaults to
            `default_reg_diag`.
        clip (bool, optional): Let `diag` be at least `default_reg_diag`.
            Defaults to `True`.
    """
    if diag is None:
        diag = epsilon
    elif clip:
        diag = B.maximum(diag, epsilon)
    return a + diag * B.eye(B.shape(a)[0], B.shape(a)[1], dtype=B.dtype(a))


@property
def log_2_pi():
    """Value of `log(2 * np.pi)`."""
    return B.log(B.array(2 * np.pi))


@property
def pi():
    """Value of `pi`."""
    return np.pi


@_dispatch(object, object)
def pw_dists2(a, b):
    """Compute the square of the pairwise Euclidean distances between design
    matrices.

    Args:
        a (design matrix, optional): First design matrix.
        b (design matrix, optional): Second design matrix. Defaults to first
            design matrix.
    """
    norms_a = B.sum(a ** 2, axis=1)[:, None]
    norms_b = B.sum(b ** 2, axis=1)[None, :]
    return norms_a + norms_b - 2 * B.matmul(a, b, tr_b=True)


@_dispatch(object)
def pw_dists2(a):
    norms = B.sum(a ** 2, axis=1)
    return norms[:, None] + norms[None, :] - 2 * B.matmul(a, a, tr_b=True)


@_dispatch(Number)
def pw_dists2(a):
    return B.array([[0.]])


@_dispatch(Number, Number)
def pw_dists2(a, b):
    return B.array([[(a - b) ** 2]])


@_dispatch([object])
def pw_dists(*args):
    """Compute the pairwise Euclidean distances between design matrices.

    Args:
        a (design matrix, optional): First design matrix.
        b (design matrix, optional): Second design matrix. Defaults to first
            design matrix.
    """
    d2 = pw_dists2(*args)
    # Clip at a bit higher than the smallest single-precision floating point
    # number.
    return B.sqrt(B.maximum(d2, B.cast(1e-30, dtype=B.dtype(d2))))


@_dispatch(Number)
def pw_dists(a):
    return B.array([[0.]])


@_dispatch(Number, Number)
def pw_dists(a, b):
    return B.array([[B.abs(a - b)]])


def is_scalar(a):
    """Check whether an object is a scalar.

    Args:
        a (tensor): Object to check.
    """
    return B.rank(a) == 0


@_dispatch(object, object)
def outer(a, b):
    """Outer product between two matrices.

    Args:
        a (matrix): First matrix.
        b (matrix, optional): Second matrix.
    """
    return B.matmul(a, b, tr_b=True)


@_dispatch(object)
def outer(a):
    return B.outer(a, a)


# Numeric type for _dispatch has to be evaluated lazily.
class PromisedNumeric(PromisedType):
    def resolve(self):
        return B._Numeric


Numeric = PromisedNumeric()


def dtype(a):
    """Get the data type of an object.

    Args:
        a (obj): Object to get data type of.
    """
    return B.array(a).dtype


def flatten(a):
    """Flatten an object.

    Args:
        a (tensor): Object to flatten.
    """
    return B.reshape(a, [-1])
