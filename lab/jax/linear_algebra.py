import logging

import jax.numpy as jnp
import jax.scipy.linalg as jsla

from . import dispatch, B, Numeric
from .custom import jax_register
from ..custom import toeplitz_solve, s_toeplitz_solve, expm, s_expm, logm, s_logm
from ..linear_algebra import _default_perm
from ..util import batch_computation

__all__ = []
log = logging.getLogger(__name__)


@dispatch
def matmul(a: Numeric, b: Numeric, tr_a=False, tr_b=False):
    a = transpose(a) if tr_a else a
    b = transpose(b) if tr_b else b
    return jnp.matmul(a, b)


@dispatch
def transpose(a: Numeric, perm=None):
    # Correctly handle special cases.
    rank_a = B.rank(a)
    if rank_a == 0:
        return a
    elif rank_a == 1 and perm is None:
        return a[None, :]

    if perm is None:
        perm = _default_perm(a)
    return jnp.transpose(a, axes=perm)


@dispatch
def trace(a: Numeric, axis1=-2, axis2=-1):
    return jnp.trace(a, axis1=axis1, axis2=axis2)


@dispatch
def kron(a: Numeric, b: Numeric):
    return jnp.kron(a, b)


@dispatch
def svd(a: Numeric, compute_uv=True):
    res = jnp.linalg.svd(a, full_matrices=False, compute_uv=compute_uv)
    return (res[0], res[1], jnp.conj(transpose(res[2]))) if compute_uv else res


@dispatch
def solve(a: Numeric, b: Numeric):
    return jnp.linalg.solve(a, b)


@dispatch
def inv(a: Numeric):
    return jnp.linalg.inv(a)


@dispatch
def det(a: Numeric):
    return jnp.linalg.det(a)


@dispatch
def logdet(a: Numeric):
    return jnp.linalg.slogdet(a)[1]


_expm = jax_register(expm, s_expm)


@dispatch
def expm(a: Numeric):
    return _expm(a)


_logm = jax_register(logm, s_logm)


@dispatch
def logm(a: Numeric):
    return _logm(a)


@dispatch
def cholesky(a: Numeric):
    return jnp.linalg.cholesky(a)


@dispatch
def cholesky_solve(a: Numeric, b: Numeric):
    return triangular_solve(transpose(a), triangular_solve(a, b), lower_a=False)


@dispatch
def triangular_solve(a: Numeric, b: Numeric, lower_a=True):
    def _triangular_solve(a_, b_):
        return jsla.solve_triangular(
            a_, b_, trans="N", lower=lower_a, check_finite=False
        )

    return batch_computation(_triangular_solve, (a, b), (2, 2))


_toeplitz_solve = jax_register(toeplitz_solve, s_toeplitz_solve)


@dispatch
def toeplitz_solve(a: Numeric, b: Numeric, c: Numeric):
    return _toeplitz_solve(a, b, c)
