import torch

from . import dispatch, Numeric
from .custom import torch_register
from ..custom import bvn_cdf, s_bvn_cdf
from ..types import TorchNumeric, NPNumeric, TorchDType, Number, Int

__all__ = []


@dispatch(Numeric)
def isnan(a):
    return torch.isnan(a)


@dispatch(TorchDType, [Int])
def zeros(dtype, *shape):
    return torch.zeros(shape, dtype=dtype)


@dispatch(TorchDType, [Int])
def ones(dtype, *shape):
    return torch.ones(shape, dtype=dtype)


@dispatch(TorchDType, Int, Int)
def eye(dtype, *shape):
    return torch.eye(shape[0], shape[1], dtype=dtype)


@dispatch(TorchDType, object, object, Int)
def linspace(dtype, a, b, num):
    return torch.linspace(a, b, num, dtype=dtype)


@dispatch(TorchDType, object, object, object)
def range(dtype, start, stop, step):
    return torch.arange(start, stop, step, dtype=dtype)


@dispatch(TorchDType, TorchNumeric)
def cast(dtype, a):
    return a.type(dtype)


@dispatch(TorchDType, {Number, NPNumeric})
def cast(dtype, a):
    return torch.tensor(a, dtype=dtype)


@dispatch(Numeric)
def identity(a):
    return a.clone()


@dispatch(Numeric)
def negative(a):
    return torch.neg(a)


@dispatch(Numeric)
def abs(a):
    return torch.abs(a)


@dispatch(Numeric)
def sign(a):
    return torch.sign(a)


@dispatch(Numeric)
def sqrt(a):
    return torch.sqrt(a)


@dispatch(Numeric)
def exp(a):
    return torch.exp(a)


@dispatch(Numeric)
def log(a):
    return torch.log(a)


@dispatch(Numeric)
def sin(a):
    return torch.sin(a)


@dispatch(Numeric)
def cos(a):
    return torch.cos(a)


@dispatch(Numeric)
def tan(a):
    return torch.tan(a)


@dispatch(Numeric)
def tanh(a):
    return torch.tanh(a)


@dispatch(Numeric, Numeric)
def add(a, b):
    return a + b


@dispatch(Numeric, Numeric)
def subtract(a, b):
    return a - b


@dispatch(Numeric, Numeric)
def multiply(a, b):
    return a * b


@dispatch(Numeric, Numeric)
def divide(a, b):
    return a / b


@dispatch(Numeric, Numeric)
def power(a, b):
    return torch.pow(a, b)


@dispatch(Numeric, Numeric)
def minimum(a, b):
    return torch.min(a, b)


@dispatch(Numeric, Numeric)
def maximum(a, b):
    return torch.max(a, b)


@dispatch(Numeric)
def min(a, axis=None):
    if axis is None:
        return torch.min(a)
    else:
        return torch.min(a, dim=axis)[0]


@dispatch(Numeric)
def max(a, axis=None):
    if axis is None:
        return torch.max(a)
    else:
        return torch.max(a, dim=axis)[0]


@dispatch(Numeric)
def sum(a, axis=None):
    if axis is None:
        return torch.sum(a)
    else:
        return torch.sum(a, dim=axis)


@dispatch(Numeric)
def mean(a, axis=None):
    if axis is None:
        return torch.mean(a)
    else:
        return torch.mean(a, dim=axis)


@dispatch(Numeric)
def std(a, axis=None):
    if axis is None:
        return torch.std(a, unbiased=False)
    else:
        return torch.std(a, dim=axis, unbiased=False)


@dispatch(Numeric)
def all(a, axis=None):
    if axis is None:
        return a.all()
    else:
        return a.all(dim=axis)


@dispatch(Numeric)
def any(a, axis=None):
    if axis is None:
        return a.any()
    else:
        return a.any(dim=axis)


@dispatch(Numeric, Numeric)
def lt(a, b):
    return torch.lt(a, b)


@dispatch(Numeric, Numeric)
def le(a, b):
    return torch.le(a, b)


@dispatch(Numeric, Numeric)
def gt(a, b):
    return torch.gt(a, b)


@dispatch(Numeric, Numeric)
def ge(a, b):
    return torch.ge(a, b)


f = torch_register(bvn_cdf, s_bvn_cdf)
dispatch(Numeric, Numeric, Numeric)(f)


@dispatch(Numeric)
def sort(a, axis=-1, descending=False):
    return torch.sort(a, dim=axis, descending=descending)[0]


@dispatch(Numeric)
def argsort(a, axis=-1, descending=False):
    return torch.argsort(a, dim=axis, descending=descending)
