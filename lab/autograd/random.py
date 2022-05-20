from plum import Union
import numpy as np

from . import dispatch
from ..types import Int, AGNumeric, AGRandomState

__all__ = []


@dispatch
def choice(
    state: AGRandomState,
    a: AGNumeric,
    n: Int,
    *,
    p: Union[AGNumeric, None] = None,
):
    # Probabilities must sum to one.
    if p is not None:
        p = p / np.sum(p, axis=0, keepdims=True)
    # Feeding `a` to `choice` will not work if `a` is higher-dimensional.
    inds = state.choice(a.shape[0], n, replace=True, p=p)
    choices = a[inds]
    return state, choices[0] if n == 1 else choices


@dispatch
def choice(a: AGNumeric, n: Int, *, p: Union[AGNumeric, None] = None):
    return choice(np.random.random.__self__, a, n, p=p)[1]
