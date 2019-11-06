import numpy as np

from . import dispatch, B, Numeric
from ..shaping import _vec_to_tril_shape_upper_perm
from ..types import Int

__all__ = []


@dispatch(Numeric)
def length(a):
    return np.size(a)


@dispatch(Numeric)
def expand_dims(a, axis=0):
    return np.expand_dims(a, axis=axis)


@dispatch(Numeric)
def squeeze(a):
    return np.squeeze(a)


@dispatch(Numeric)
def diag(a):
    return np.diag(a)


@dispatch(Numeric)
def vec_to_tril(a):
    if B.rank(a) != 1:
        raise ValueError('Input must be rank 1.')
    m, upper, perm = _vec_to_tril_shape_upper_perm(a)
    a = np.concatenate((a, np.zeros(upper, dtype=a.dtype)))
    return np.reshape(a[perm], (m, m))


@dispatch(Numeric)
def tril_to_vec(a):
    if B.rank(a) != 2:
        raise ValueError('Input must be rank 2.')
    n, m = B.shape(a)
    if n != m:
        raise ValueError('Input must be square.')
    return a[np.tril_indices(n)]


@dispatch([Numeric])
def stack(*elements, **kw_args):
    return np.stack(elements, axis=kw_args.get('axis', 0))


@dispatch(Numeric)
def unstack(a, axis=0):
    out = np.split(a, np.arange(1, a.shape[axis]), axis)
    return [x.squeeze(axis=axis) for x in out]


@dispatch(Numeric, [Int])
def reshape(a, *shape):
    return np.reshape(a, shape)


@dispatch([Numeric])
def concat(*elements, **kw_args):
    return np.concatenate(elements, axis=kw_args.get('axis', 0))


@dispatch(Numeric, [Int])
def tile(a, *repeats):
    return np.tile(a, repeats)


@dispatch(Numeric, object)
def take(a, indices_or_mask, axis=0):
    if B.rank(indices_or_mask) != 1:
        raise ValueError('Indices or mask must be rank 1.')

    # Put axis `axis` first.
    if axis > 0:
        # Create a permutation to switch `axis` and `0`.
        perm = list(range(B.rank(a)))
        perm[0], perm[axis] = perm[axis], perm[0]
        a = np.transpose(a, perm)

    # Take the relevant part.
    a = a[indices_or_mask, ...]

    # Put axis `axis` back again.
    if axis > 0:
        a = np.transpose(a, perm)

    return a
