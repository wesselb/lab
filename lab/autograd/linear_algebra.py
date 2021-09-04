import logging
from typing import Union, Optional

import autograd.numpy as anp
import autograd.scipy.linalg as asla

from . import dispatch, B, Numeric
from .custom import autograd_register
from ..custom import toeplitz_solve, s_toeplitz_solve, expm, s_expm, logm, s_logm
from ..linear_algebra import _default_perm
from ..types import Int
from ..util import batch_computation, resolve_axis

__all__ = []
log = logging.getLogger(__name__)


@dispatch
def matmul(a: Numeric, b: Numeric, tr_a: bool = False, tr_b: bool = False):
    a = transpose(a) if tr_a else a
    b = transpose(b) if tr_b else b
    return anp.matmul(a, b)


@dispatch
def transpose(a: Numeric, perm: Optional[Union[tuple, list]] = None):
    # Correctly handle special cases.
    rank_a = B.rank(a)
    if rank_a == 0:
        return a
    elif rank_a == 1 and perm is None:
        return a[None, :]

    if perm is None:
        perm = _default_perm(a)
    return anp.transpose(a, axes=perm)


@dispatch
def trace(a: Numeric, axis1: Int = -2, axis2: Int = -1):
    axis1 = resolve_axis(a, axis1)
    axis2 = resolve_axis(a, axis2)

    if axis1 == axis2:
        raise ValueError("Keyword arguments `axis1` and `axis2` cannot be the same.")

    # AutoGrad does not support the `axis1` and `axis2` arguments...

    # Order the axis as `axis1 < axis`.
    if axis2 < axis1:
        axis1, axis2 = axis2, axis1

    # Bring the trace axes forward.
    if (axis1, axis2) != (0, 1):
        perm = [axis1, axis2]
        perm += [i for i in range(B.rank(a)) if i != axis1 and i != axis2]
        a = anp.transpose(a, axes=perm)

    return anp.trace(a)


@dispatch
def svd(a: Numeric, compute_uv: bool = True):
    res = anp.linalg.svd(a, full_matrices=False, compute_uv=compute_uv)
    return (res[0], res[1], anp.conj(transpose(res[2]))) if compute_uv else res


@dispatch
def eig(a: Numeric, compute_eigvecs: bool = True):  # pragma: no cover
    raise NotImplementedError("Function `quantile` is not available for AutoGrad.")


@dispatch
def solve(a: Numeric, b: Numeric):
    return anp.linalg.solve(a, b)


@dispatch
def inv(a: Numeric):
    return anp.linalg.inv(a)


@dispatch
def det(a: Numeric):
    return anp.linalg.det(a)


@dispatch
def logdet(a: Numeric):
    return anp.linalg.slogdet(a)[1]


_expm = autograd_register(expm, s_expm)


@dispatch
def expm(a: Numeric):
    return _expm(a)


_logm = autograd_register(logm, s_logm)


@dispatch
def logm(a: Numeric):
    return _logm(a)


@dispatch
def _cholesky(a: Numeric):
    return anp.linalg.cholesky(a)


@dispatch
def cholesky_solve(a: Numeric, b: Numeric):
    return triangular_solve(transpose(a), triangular_solve(a, b), lower_a=False)


@dispatch
def triangular_solve(a: Numeric, b: Numeric, lower_a: bool = True):
    def _triangular_solve(a_, b_):
        return asla.solve_triangular(
            a_, b_, trans="N", lower=lower_a, check_finite=False
        )

    return batch_computation(_triangular_solve, (a, b), (2, 2))


_toeplitz_solve = autograd_register(toeplitz_solve, s_toeplitz_solve)


@dispatch
def toeplitz_solve(a: Numeric, b: Numeric, c: Numeric):
    return _toeplitz_solve(a, b, c)
