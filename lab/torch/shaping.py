import numpy as np
import torch

from . import dispatch, B, Numeric
from ..shaping import _vec_to_tril_shape_upper_perm
from ..types import Int

__all__ = []


@dispatch(Numeric)
def length(a):
    return a.numel()


@dispatch(Numeric)
def expand_dims(a, axis=0):
    return torch.unsqueeze(a, dim=axis)


@dispatch(Numeric)
def squeeze(a):
    return torch.squeeze(a)


@dispatch(Numeric)
def diag(a):
    return torch.diag(a)


@dispatch(Numeric)
def vec_to_tril(a):
    if B.rank(a) != 1:
        raise ValueError('Input must be rank 1.')
    m, upper, perm = _vec_to_tril_shape_upper_perm(a)
    a = torch.cat((a, torch.zeros(upper, dtype=a.dtype)))
    return torch.reshape(a[perm], (m, m))


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
    return torch.stack(elements, dim=kw_args.get('axis', 0))


@dispatch(Numeric)
def unstack(a, axis=0):
    return torch.unbind(a, dim=axis)


@dispatch(Numeric, [Int])
def reshape(a, *shape):
    return torch.reshape(a, shape=shape)


@dispatch([Numeric])
def concat(*elements, **kw_args):
    return torch.cat(elements, dim=kw_args.get('axis', 0))


@dispatch(Numeric, [Int])
def tile(a, *repeats):
    return a.repeat(*repeats)


@dispatch(Numeric, object)
def take(a, indices_or_mask, axis=0):
    if B.rank(indices_or_mask) != 1:
        raise ValueError('Indices or mask must be rank 1.')

    # Put axis `axis` first.
    if axis > 0:
        a = torch.transpose(a, 0, axis)

    # Take the relevant part.
    a = a[indices_or_mask, ...]

    # Put axis `axis` back again.
    if axis > 0:
        a = torch.transpose(a, 0, axis)

    return a
