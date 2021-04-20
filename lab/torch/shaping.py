import torch

from . import dispatch, Numeric
from ..types import Int

__all__ = []


@dispatch
def length(a: Numeric):
    return a.numel()


@dispatch
def expand_dims(a: Numeric, axis=0):
    return torch.unsqueeze(a, dim=axis)


@dispatch
def squeeze(a: Numeric):
    return torch.squeeze(a)


@dispatch
def diag(a: Numeric):
    return torch.diag(a)


@dispatch
def diag_extract(a: Numeric):
    return torch.diagonal(a, dim1=-2, dim2=-1)


@dispatch
def diag_construct(a: Numeric):
    return torch.diag_embed(a, dim1=-2, dim2=-1)


@dispatch
def stack(*elements: Numeric, axis=0):
    return torch.stack(elements, dim=axis)


@dispatch
def unstack(a: Numeric, axis=0):
    return torch.unbind(a, dim=axis)


@dispatch
def reshape(a: Numeric, *shape: Int):
    return torch.reshape(a, shape=shape)


@dispatch
def concat(*elements: Numeric, axis=0):
    return torch.cat(elements, dim=axis)


@dispatch
def tile(a: Numeric, *repeats: Int):
    return a.repeat(*repeats)
