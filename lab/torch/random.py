import torch

from . import B, dispatch
from ..types import TorchNumeric, TorchDType, Int

__all__ = []


@dispatch
def rand(dtype: TorchDType, *shape: Int):
    return torch.rand(shape, dtype=dtype, device=B.Device.active_name)


@dispatch
def randn(dtype: TorchDType, *shape: Int):
    return torch.randn(shape, dtype=dtype, device=B.Device.active_name)


@dispatch
def choice(a: TorchNumeric, n: Int):
    choices = a[torch.randint(a.shape[0], (n,))]
    return choices[0] if n == 1 else choices
