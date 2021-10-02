import logging
from collections import namedtuple
from functools import reduce

import numpy as np
import scipy.linalg as sla

TensorDescription = namedtuple("TensorDescription", "shape dtype")
"""namedtuple: Description of a tensor in terms of the tensor's shape and data type."""


def promote_dtype_of_tensors(*xs):
    """Promote the data types of a number of tensors.

    Args:
        *xs (tensor): Tensors to take data types of and then promote those data types.

    Returns:
        dtype: Promoted data type.
    """
    return reduce(np.promote_types, [x.dtype for x in xs])


try:
    # noinspection PyUnresolvedReferences
    from .bvn_cdf import bvn_cdf as bvn_cdf_, s_bvn_cdf as s_bvn_cdf_

    def i_bvn_cdf(a, b, c):
        if a.shape != b.shape or a.shape != c.shape:
            raise ValueError("Shapes of the inputs to `bvn_cdf` must all be equal.")
        return TensorDescription(a.shape, promote_dtype_of_tensors(a, b, c))

    def i_s_bvn_cdf(s_y, y, a, b, c):
        dtype = promote_dtype_of_tensors(s_y, y, a, b, c)
        return (
            TensorDescription(a.shape, dtype),
            TensorDescription(b.shape, dtype),
            TensorDescription(c.shape, dtype),
        )


except ImportError:  # pragma: no cover

    def bvn_cdf_(*args, **kw_args):
        raise RuntimeError(
            "bvn_cdf was not compiled. Please try to reinstall LAB with `gfortran` "
            "available."
        )

    def i_bvn_cdf(*args, **kw_args):
        raise RuntimeError(
            "bvn_cdf was not compiled. Please try to reinstall LAB with `gfortran` "
            "available."
        )

    def s_bvn_cdf_(*args, **kw_args):
        raise RuntimeError(
            "bvn_cdf was not compiled. Please try to reinstall LAB with `gfortran` "
            "available."
        )

    def i_s_bvn_cdf(*args, **kw_args):
        raise RuntimeError(
            "bvn_cdf was not compiled. Please try to reinstall LAB with `gfortran` "
            "available."
        )


__all__ = [
    "toeplitz_solve",
    "i_toeplitz_solve",
    "s_toeplitz_solve",
    "i_s_toeplitz_solve",
    "bvn_cdf",
    "i_bvn_cdf",
    "s_bvn_cdf",
    "i_s_bvn_cdf",
    "expm",
    "i_expm",
    "s_expm",
    "i_s_expm",
    "logm",
    "i_logm",
    "s_logm",
    "i_s_logm",
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
    # For some reason, `sla.solve_toeplitz` sometimes fails with a `ValueError`, saying
    # that the buffer source array is read-only. We resolve this issue by copying the
    # inputs....
    # TODO: Resolve this properly.
    a = np.copy(a)
    b = np.copy(b)
    c = np.copy(c)
    res_dtype = promote_dtype_of_tensors(a, b, c)
    row = np.concatenate((a[:1], b))  # First row of the Toeplitz matrix
    return sla.solve_toeplitz((a, row), c).astype(res_dtype)


def i_toeplitz_solve(a, b, c):
    return TensorDescription(c.shape, promote_dtype_of_tensors(a, b, c))


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


def i_s_toeplitz_solve(s_y, y, a, b, c):
    dtype = promote_dtype_of_tensors(s_y, y, a, b, c)
    return (
        TensorDescription(a.shape, dtype),
        TensorDescription(b.shape, dtype),
        TensorDescription(c.shape, dtype),
    )


def bvn_cdf(a, b, c):
    # We do not directly use `bvn_cdf_` to not have `inspect.signature` fail, which
    # does not work for `bvn_cdf_`. Moreover, we need to ensure that the function
    # runs on `float64s`.
    res_dtype = reduce(np.promote_types, [x.dtype for x in (a, b, c)])
    res = bvn_cdf_(a.astype(np.float64), b.astype(np.float64), c.astype(np.float64))
    return res.astype(res_dtype)


def s_bvn_cdf(s_y, y, a, b, c):
    res_dtype = reduce(np.promote_types, [x.dtype for x in (s_y, y, a, b, c)])
    res = s_bvn_cdf_(
        s_y.astype(np.float64),
        y.astype(np.float64),
        a.astype(np.float64),
        b.astype(np.float64),
        c.astype(np.float64),
    )
    return tuple(x.astype(res_dtype) for x in res)


def expm(a):
    return sla.expm(a)


def i_expm(a):
    return TensorDescription(a.shape, a.dtype)


def s_expm(s_y, y, a):
    return sla.expm_frechet(a, s_y.T, compute_expm=False).T


def i_s_expm(s_y, y, a):
    return TensorDescription(a.shape, promote_dtype_of_tensors(s_y, y, a))


def logm(a):
    return sla.logm(a)


def i_logm(a):
    return TensorDescription(a.shape, a.dtype)


def s_logm(a):  # pragma: no cover
    raise NotImplementedError(
        "The derivative for the matrix logarithm is current not implemented."
    )


def i_s_logm(s_y, y, a):  # pragma: no cover
    raise NotImplementedError(
        "The derivative for the matrix logarithm is current not implemented."
    )
