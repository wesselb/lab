import logging

import numpy as np
import scipy.linalg as sla
from scipy.stats import norm

from .bvn_cdf import bvn_cdf

__all__ = ['toeplitz_solve', 's_toeplitz_solve',
           'bvn_cdf', 's_bvn_cdf']

log = logging.getLogger(__name__)


def _mm(a, b):
    """Short hand for `np.matmul`.

    Args:
        a (tensor): First tensor in product.
        b (tensor): Second tensor in product.

    Return:
        tensor: Matrix product of `a` and `b`.
    """

    return np.matmul(a, b)


def _t(a):
    """Transpose `a`, correctly handling the case where `a` is rank one.

    Args:
        a (tensor): Tensor to transpose.

    Returns:
        tensor: Transposition of `a`.
    """
    if a.ndim == 1:
        return a[None, :]
    else:
        return np.transpose(a)


def _uprank(a):
    """Get `a` as a rank-two tensor, correctly handling the case where `a` is
    rank one.

    Args:
        a (tensor): Tensor to get as a rank-two tensor.

    Returns:
        tensor: `a` as a rank-two vector.
    """
    if a.ndim == 1:
        return a[:, None]
    else:
        return a


def toeplitz_solve(a, b, c):
    row = np.concatenate((a[:1], b))  # First row of the Toeplitz matrix.
    return sla.solve_toeplitz((a, row), c)


def s_toeplitz_solve(s_y, y, a, b, c):
    # Compute `a` and `b` to get the transpose of the Toeplitz matrix.
    a_t = np.concatenate((a[:1], b))
    b_t = a[1:]

    # Compute the sensitivity w.r.t `c`.
    s_c = toeplitz_solve(a_t, b_t, s_y)

    # Compute the sensitivity w.r.t. the transposed inverse of the Toeplitz
    # matrix.
    s_inv = -_mm(_uprank(s_c), _t(y))

    # Finally, compute the sensitivities w.r.t. `a` and `c`.
    n = a.shape[0]
    s_a = np.array([s_inv.diagonal(-i).sum() for i in range(n)])
    s_b = np.array([s_inv.diagonal(i).sum() for i in range(1, n)])

    return s_a, s_b, s_c


def s_bvn_cdf(s_y, y, a, b, c):
    q = np.sqrt(1 - c ** 2)

    # Compute the densities.
    pdfs = 1 / (2 * np.pi * q) * \
           np.exp(-(a ** 2 - 2 * c * a * b + b ** 2) /
                  (2 * (1 - c ** 2)))

    # Compute sensitivities.
    s_a = s_y * norm.pdf(a) * norm.cdf(b, c * a, q)
    s_b = s_y * norm.pdf(b) * norm.cdf(a, c * b, q)
    s_c = s_y * pdfs

    return s_a, s_b, s_c
