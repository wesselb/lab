# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import torch

from . import dispatch, B
from ..shaping import _vec_to_tril_shape
from ..types import TorchNumeric, TorchDimension

__all__ = []


@dispatch(TorchNumeric)
def length(a):
    return a.numel()


@dispatch(TorchNumeric)
def expand_dims(a, axis=0):
    return torch.unsqueeze(a, dim=axis)


@dispatch(TorchNumeric)
def squeeze(a):
    return torch.squeeze(a)


@dispatch(TorchNumeric)
def diag(a):
    return torch.diag(a)


@dispatch(TorchNumeric)
def vec_to_tril(a):
    if B.rank(a) != 1:
        raise ValueError('Input must be rank 1.')
    m = _vec_to_tril_shape(a)
    out = torch.zeros(m, m, dtype=a.dtype)
    out[np.tril_indices(m)] = a
    return out


@dispatch(TorchNumeric)
def tril_to_vec(a):
    if B.rank(a) != 2:
        raise ValueError('Input must be rank 2.')
    n, m = B.shape(a)
    if n != m:
        raise ValueError('Input must be square.')
    return a[np.tril_indices(n)]


@dispatch([TorchNumeric])
def stack(*elements, **kw_args):
    return torch.stack(elements, dim=kw_args.get('axis', 0))


@dispatch(TorchNumeric)
def unstack(a, axis=0):
    return torch.unbind(a, dim=axis)


@dispatch(TorchNumeric, [TorchDimension])
def reshape(a, *shape):
    return torch.reshape(a, shape=shape)


@dispatch([TorchNumeric])
def concat(*elements, **kw_args):
    return torch.cat(elements, dim=kw_args.get('axis', 0))


@dispatch(TorchNumeric, [TorchDimension])
def tile(a, *repeats):
    return a.repeat(*repeats)


@dispatch(TorchNumeric, object)
def take(a, indices_or_mask, axis=0):
    # Put axis `axis` first.
    if axis > 0:
        a = torch.transpose(a, 0, axis)

    # Take the relevant part.
    a = a[indices_or_mask, ...]

    # Put axis `axis` back again.
    if axis > 0:
        a = torch.transpose(a, 0, axis)

    return a
