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


@dispatch([Numeric])
def stack(*elements, **kw_args):
    return torch.stack(elements, dim=kw_args.get("axis", 0))


@dispatch(Numeric)
def unstack(a, axis=0):
    return torch.unbind(a, dim=axis)


@dispatch(Numeric, [Int])
def reshape(a, *shape):
    return torch.reshape(a, shape=shape)


@dispatch([Numeric])
def concat(*elements, **kw_args):
    return torch.cat(elements, dim=kw_args.get("axis", 0))


@dispatch(Numeric, [Int])
def tile(a, *repeats):
    return a.repeat(*repeats)
