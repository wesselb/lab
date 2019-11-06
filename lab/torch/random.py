import torch

from . import dispatch
from ..types import TorchNumeric, TorchDType, Int

__all__ = []


@dispatch(TorchDType, [Int])
def rand(dtype, *shape):
    return torch.rand(shape, dtype=dtype)


@dispatch(TorchDType, [Int])
def randn(dtype, *shape):
    return torch.randn(shape, dtype=dtype)


@dispatch(TorchNumeric, Int)
def choice(a, n):
    choices = a[torch.randint(a.shape[0], (n,))]
    return choices[0] if n == 1 else choices
