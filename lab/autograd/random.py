import numpy as np

from . import dispatch
from ..types import Int, AGNumeric

__all__ = []


@dispatch(AGNumeric, Int)
def choice(a, n):
    inds = np.random.choice(a.shape[0], n, replace=True)
    choices = a[inds]
    return choices[0] if n == 1 else choices
