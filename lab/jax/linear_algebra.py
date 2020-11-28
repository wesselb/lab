import logging

import jax.numpy as jnp
import jax.scipy.linalg as jsla

from . import dispatch, B, Numeric
from .custom import jax_register
from ..custom import (
    toeplitz_solve, s_toeplitz_solve,
    expm, s_expm,
    logm, s_logm
)
from ..linear_algebra import _default_perm
from ..util import batch_computation

__all__ = []
log = logging.getLogger(__name__)


@dispatch(Numeric, Numeric)
def matmul(a, b, tr_a=False, tr_b=False):
    a = transpose(a) if tr_a else a
    b = transpose(b) if tr_b else b
    return jnp.matmul(a, b)


@dispatch(Numeric)
def transpose(a, perm=None):
    # Correctly handle special cases.
    rank_a = B.rank(a)
    if rank_a == 0:
        return a
    elif rank_a == 1 and perm is None:
        return a[None, :]

    if perm is None:
        perm = _default_perm(a)
    return jnp.transpose(a, axes=perm)


@dispatch(Numeric)
def trace(a, axis1=0, axis2=1):
    return jnp.trace(a, axis1=axis1, axis2=axis2)


@dispatch(Numeric, Numeric)
def kron(a, b):
    return jnp.kron(a, b)


@dispatch(Numeric)
def svd(a, compute_uv=True):
    res = jnp.linalg.svd(a, full_matrices=False, compute_uv=compute_uv)
    return (res[0], res[1], jnp.conj(transpose(res[2]))) if compute_uv else res


@dispatch(Numeric, Numeric)
def solve(a, b):
    return jnp.linalg.solve(a, b)


@dispatch(Numeric)
def inv(a):
    return jnp.linalg.inv(a)


@dispatch(Numeric)
def det(a):
    return jnp.linalg.det(a)


@dispatch(Numeric)
def logdet(a):
    return jnp.linalg.slogdet(a)[1]


f = jax_register(expm, s_expm)
dispatch(Numeric)(f)

f = jax_register(logm, s_logm)
dispatch(Numeric)(f)


@dispatch(Numeric)
def cholesky(a):
    return jnp.linalg.cholesky(a)


@dispatch(Numeric, Numeric)
def cholesky_solve(a, b):
    return triangular_solve(transpose(a), triangular_solve(a, b), lower_a=False)


@dispatch(Numeric, Numeric)
def triangular_solve(a, b, lower_a=True):
    def _triangular_solve(a_, b_):
        return jsla.solve_triangular(a_, b_,
                                     trans='N',
                                     lower=lower_a,
                                     check_finite=False)

    return batch_computation(_triangular_solve, (a, b), (2, 2))


f = jax_register(toeplitz_solve, s_toeplitz_solve)
dispatch(Numeric, Numeric, Numeric)(f)
