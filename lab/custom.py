import logging

import numpy as np
import scipy.linalg as sla

# noinspection PyUnresolvedReferences
try:
    from .bvn_cdf import bvn_cdf as bvn_cdf_, s_bvn_cdf
except ImportError:  # pragma: no cover

    def bvn_cdf_(*args, **kw_args):
        raise RuntimeError(
            "bvn_cdf was not compiled. Please try to reinstall LAB with `gfortran` "
            "available."
        )

    def s_bvn_cdf(*args, **kw_args):
        raise RuntimeError(
            "bvn_cdf was not compiled. Please try to reinstall LAB with `gfortran` "
            "available."
        )


__all__ = [
    "toeplitz_solve",
    "s_toeplitz_solve",
    "bvn_cdf",
    "s_bvn_cdf",
    "expm",
    "s_expm",
    "logm",
    "s_logm",
]

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


def bvn_cdf(a, b, c):
    # We do not directly use `bvn_cdf_` to not have `inspect.signature` fail, which
    # does not work for `bvn_cdf_`.
    return bvn_cdf_(a, b, c)


def expm(a):
    return sla.expm(a)


def s_expm(s_y, y, a):
    return sla.expm_frechet(a, s_y.T, compute_expm=False).T


def logm(a):
    return sla.logm(a)


def s_logm(a):  # pragma: no cover
    raise NotImplementedError(
        "The derivative for the matrix logarithm is current not implemented."
    )
