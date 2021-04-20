from functools import wraps

import torch
from plum import convert, Dispatcher

from . import B

__all__ = ["torch_register", "as_torch"]

_dispatch = Dispatcher()


@_dispatch
def as_torch(x: B.Numeric, grad=False):
    """Convert object to PyTorch.

    Args:
        x (object): Object to convert.
        grad (bool, optional): Requires gradient. Defaults to `False`.

    Returns:
        object: `x` as a PyTorch object.
    """
    dtype = convert(B.dtype(x), B.TorchDType)
    return torch.tensor(x, dtype=dtype, requires_grad=grad)


@_dispatch
def as_torch(xs: tuple, grad=False):
    return tuple([as_torch(x, grad=grad) for x in xs])


def torch_register(f, s_f):
    """Register a function and its sensitivity for PyTorch.

    Args:
        f (function): Function to register.
        s_f (function): Sensitivity of `f`.

    Returns:
        function: PyTorch primitive.
    """

    # Create a custom PyTorch function.
    class Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            y = f(*B.to_numpy(args))
            ctx.save_for_backward(as_torch(y), *args)
            return as_torch(y)

        @staticmethod
        def backward(ctx, s_y):  # pragma: no cover
            # The profiler does not catch that this is tested.
            y = ctx.saved_tensors[0]
            args = ctx.saved_tensors[1:]
            return as_torch(s_f(s_y.numpy(), y.numpy(), *B.to_numpy(args)))

    # Wrap it to preserve the function name.

    @wraps(f)
    def f_wrapped(*args, **kw_args):
        return Function.apply(*args, **kw_args)

    return f_wrapped
