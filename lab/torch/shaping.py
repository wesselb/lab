import torch

from . import dispatch, Numeric
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
def diag_extract(a):
    return torch.diagonal(a, dim1=-2, dim2=-1)


@dispatch(Numeric)
def diag_construct(a):
    return torch.diag_embed(a, dim1=-2, dim2=-1)


@dispatch([Numeric])
def stack(*elements, axis=0):
    return torch.stack(elements, dim=axis)


@dispatch(Numeric)
def unstack(a, axis=0):
    return torch.unbind(a, dim=axis)


@dispatch(Numeric, [Int])
def reshape(a, *shape):
    return torch.reshape(a, shape=shape)


@dispatch([Numeric])
def concat(*elements, axis=0):
    return torch.cat(elements, dim=axis)


@dispatch(Numeric, [Int])
def tile(a, *repeats):
    return a.repeat(*repeats)
