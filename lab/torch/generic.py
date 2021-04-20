import torch
from plum import Union

from . import B, dispatch, Numeric
from .custom import torch_register
from ..custom import bvn_cdf, s_bvn_cdf
from ..shape import Dimension
from ..types import TorchNumeric, NPNumeric, TorchDType, Number, Int

__all__ = []


@dispatch
def isnan(a: Numeric):
    return torch.isnan(a)


@dispatch
def device(a: TorchNumeric):
    return str(a.device)


@dispatch
def move_to_active_device(a: TorchNumeric):
    return a.to(B.Device.active_name)


@dispatch
def zeros(dtype: TorchDType, *shape: Int):
    return torch.zeros(shape, dtype=dtype, device=B.Device.active_name)


@dispatch
def ones(dtype: TorchDType, *shape: Int):
    return torch.ones(shape, dtype=dtype, device=B.Device.active_name)


@dispatch
def _eye2(dtype: TorchDType, *shape: Int):
    return torch.eye(shape[0], shape[1], dtype=dtype, device=B.Device.active_name)


@dispatch
def linspace(dtype: TorchDType, a, b, num: Int):
    return torch.linspace(a, b, num, dtype=dtype, device=B.Device.active_name)


@dispatch
def range(dtype: TorchDType, start, stop, step):
    return torch.arange(start, stop, step, dtype=dtype, device=B.Device.active_name)


@dispatch
def cast(dtype: TorchDType, a: TorchNumeric):
    return a.type(dtype)


@dispatch
def cast(dtype: TorchDType, a: Union[Number, NPNumeric]):
    return torch.tensor(a, dtype=dtype, device=B.Device.active_name)


@dispatch
def cast(dtype: TorchDType, a: Dimension):
    # A dimension may automatically unwrap to a PyTorch tensor.
    return cast(dtype, a)


@dispatch
def identity(a: Numeric):
    return 1 * a


@dispatch
def negative(a: Numeric):
    return torch.neg(a)


@dispatch
def abs(a: Numeric):
    return torch.abs(a)


@dispatch
def sign(a: Numeric):
    return torch.sign(a)


@dispatch
def sqrt(a: Numeric):
    return torch.sqrt(a)


@dispatch
def exp(a: Numeric):
    return torch.exp(a)


@dispatch
def log(a: Numeric):
    return torch.log(a)


@dispatch
def sin(a: Numeric):
    return torch.sin(a)


@dispatch
def cos(a: Numeric):
    return torch.cos(a)


@dispatch
def tan(a: Numeric):
    return torch.tan(a)


@dispatch
def tanh(a: Numeric):
    return torch.tanh(a)


@dispatch
def erf(a: Numeric):
    return torch.erf(a)


@dispatch
def add(a: Numeric, b: Numeric):
    return a + b


@dispatch
def subtract(a: Numeric, b: Numeric):
    return a - b


@dispatch
def multiply(a: Numeric, b: Numeric):
    return a * b


@dispatch
def divide(a: Numeric, b: Numeric):
    return a / b


@dispatch
def power(a: Numeric, b: Numeric):
    return torch.pow(a, b)


@dispatch
def minimum(a: Numeric, b: Numeric):
    return torch.min(a, b)


@dispatch
def maximum(a: Numeric, b: Numeric):
    return torch.max(a, b)


@dispatch
def min(a: Numeric, axis=None):
    if axis is None:
        return torch.min(a)
    else:
        return torch.min(a, dim=axis)[0]


@dispatch
def max(a: Numeric, axis=None):
    if axis is None:
        return torch.max(a)
    else:
        return torch.max(a, dim=axis)[0]


@dispatch
def sum(a: Numeric, axis=None):
    if axis is None:
        return torch.sum(a)
    else:
        return torch.sum(a, dim=axis)


@dispatch
def mean(a: Numeric, axis=None):
    if axis is None:
        return torch.mean(a)
    else:
        return torch.mean(a, dim=axis)


@dispatch
def std(a: Numeric, axis=None):
    if axis is None:
        return torch.std(a, unbiased=False)
    else:
        return torch.std(a, dim=axis, unbiased=False)


@dispatch
def all(a: Numeric, axis=None):
    if axis is None:
        return a.all()
    else:
        return a.all(dim=axis)


@dispatch
def any(a: Numeric, axis=None):
    if axis is None:
        return a.any()
    else:
        return a.any(dim=axis)


@dispatch
def lt(a: Numeric, b: Numeric):
    return torch.lt(a, b)


@dispatch
def le(a: Numeric, b: Numeric):
    return torch.le(a, b)


@dispatch
def gt(a: Numeric, b: Numeric):
    return torch.gt(a, b)


@dispatch
def ge(a: Numeric, b: Numeric):
    return torch.ge(a, b)


_bvn_cdf = torch_register(bvn_cdf, s_bvn_cdf)


@dispatch
def bvn_cdf(a: Numeric, b: Numeric, c: Numeric):
    return _bvn_cdf(a, b, c)


@dispatch
def sort(a: Numeric, axis=-1, descending=False):
    return torch.sort(a, dim=axis, descending=descending)[0]


@dispatch
def argsort(a: Numeric, axis=-1, descending=False):
    return torch.argsort(a, dim=axis, descending=descending)
