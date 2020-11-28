import numpy as np

from . import dispatch
from ..types import Int, JaxNumeric

__all__ = []


# Since Jax data types are NumPy data types, we do not implement `rand` and `randn`:
# these simply dispatch to the NumPy functions; the call to `B.cast` will take care
# of the conversion to a Jax data type.


@dispatch(JaxNumeric, Int)
def choice(a, n):
    inds = np.random.choice(a.shape[0], n, replace=True)
    choices = a[inds]
    return choices[0] if n == 1 else choices
