# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from . import B

__all__ = ['default_reg_diag', 'rank', 'reg', 'log_2_pi', 'pi', 'pw_dists2']

default_reg_diag = 1e-12  #: Magnitude of diagonal to regularise matrices with.


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
        diag = default_reg_diag
    elif clip:
        diag = B.maximum(diag, default_reg_diag)
    return a + diag * B.eye(B.shape(a)[0],
                            B.shape(a)[1], dtype=a.dtype)


@property
def log_2_pi():
    """Value of `log(2 * np.pi)`."""
    return B.log(B.array(2 * np.pi))


@property
def pi():
    """Value of `pi`."""
    return np.pi


def pw_dists2(a, b=None):
    """Compute the square of the pairwise Euclidean distances between design
    matrices.

    Args:
        a (design matrix, optional): First design matrix.
        b (design matrix, optional): Second design matrix. Defaults to first
            design matrix.
    """
    if b is None:
        norms = B.sum(a ** 2, axis=1)
        return norms[:, None] + norms[None, :] - 2 * B.matmul(a, a, tr_b=True)
    else:
        norms_a = B.sum(a ** 2, axis=1)[:, None]
        norms_b = B.sum(b ** 2, axis=1)[None, :]
        return norms_a + norms_b - 2 * B.matmul(a, b, tr_b=True)


def pw_dists(a, b=None, dists2=False):
    """Compute the pairwise Euclidean distances between design matrices.

    Args:
        a (design matrix, optional): First design matrix.
        b (design matrix, optional): Second design matrix. Defaults to first
            design matrix.
        dists2 (bool, optional): Also return squared distances. Defaults to
            `false`.
    """
    d2 = pw_dists2(a, b)
    # Adding 1e-8 here is highly suboptimal, but unfortunately required to
    # ensure stable gradients.
    d = B.sqrt(d2 + B.default_reg_diag)
    if dists2:
        return d, d2
    else:
        return d


def is_scalar(a):
    """Check whether an object is a scalar.

    Args:
        a (tensor): Object to check.
    """
    return B.rank(a) == 0
