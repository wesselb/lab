import autograd.numpy as anp
import numpy as np

from . import dispatch, B
from ..types import Int, AGNumeric, AGRandomState

__all__ = []


@dispatch
def randcat(state: AGRandomState, p: AGNumeric, n: Int):
    # Probabilities must sum to one.
    p = p / anp.sum(p, axis=-1, keepdims=True)
    # Perform sampling routine.
    cdf = anp.cumsum(p, axis=-1)
    u = state.rand(n, *p.shape[:-1])
    inds = anp.sum(u[..., None] < cdf[None], axis=-1) - 1
    # Be sure to return the right data type.
    return state, B.cast(B.dtype_int(p), inds)


@dispatch
def randcat(p: AGNumeric, *shape: Int):
    return randcat(np.random.random.__self__, p, *shape)[1]
