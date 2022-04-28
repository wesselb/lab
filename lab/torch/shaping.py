import numpy as np
import torch
from plum import Union, convert

from . import dispatch, Numeric, TorchNumeric
from ..types import Int, NPDType, issubdtype

__all__ = []


@dispatch
def length(a: Numeric):
    return a.numel()


@dispatch
def _expand_dims(a: Numeric, axis: Int = 0):
    return torch.unsqueeze(a, dim=axis)


@dispatch
def squeeze(a: Numeric, axis: Union[Int, None] = None):
    if axis is None:
        return torch.squeeze(a)
    else:
        return torch.squeeze(a, dim=axis)


@dispatch
def broadcast_to(a: Numeric, *shape: Int):
    return torch.broadcast_to(a, shape)


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
def stack(*elements: Numeric, axis: Int = 0):
    return torch.stack(elements, dim=axis)


@dispatch
def _unstack(a: Numeric, axis: Int = 0):
    return torch.unbind(a, dim=axis)


@dispatch
def reshape(a: Numeric, *shape: Int):
    return torch.reshape(a, shape=shape)


@dispatch
def concat(*elements: Numeric, axis: Int = 0):
    return torch.cat(elements, dim=axis)


@dispatch
def tile(a: Numeric, *repeats: Int):
    return a.repeat(*repeats)


@dispatch
def _take_convert(indices_or_mask: Union[list, tuple]):
    return indices_or_mask


@dispatch
def _take_convert(indices_or_mask: TorchNumeric):
    if issubdtype(convert(indices_or_mask.dtype, NPDType), np.integer):
        # Indices must be on the CPU and `int64`s!
        return indices_or_mask.cpu().type(torch.int64)
    else:
        return indices_or_mask
