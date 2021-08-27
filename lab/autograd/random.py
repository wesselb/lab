import numpy as np

from . import dispatch
from ..types import Int, AGNumeric, AGRandomState

__all__ = []


@dispatch
def choice(state: AGRandomState, a: AGNumeric, n: Int):
    inds = state.choice(a.shape[0], n, replace=True)
    choices = a[inds]
    return state, choices[0] if n == 1 else choices


@dispatch
def choice(a: AGNumeric, n: Int):
    return choice(np.random.random.__self__, a, n)[1]
