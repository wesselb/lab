import logging
from itertools import product

import jax.numpy as jnp
import numpy as np
import plum
import pytest
import tensorflow as tf
import torch
from autograd.core import VJPNode, getval
from autograd.tracer import trace_stack, new_box
from plum import Dispatcher, Union

import lab as B
from lab.shape import Shape, Dimension, unwrap_dimension

__all__ = [
    "check_lazy_shapes",
    "autograd_box",
    "to_np",
    "allclose",
    "check_function",
    "Tensor",
    "PositiveTensor",
    "BoolTensor",
    "NaNTensor",
    "Matrix",
    "PSD",
    "PSDTriangular",
    "Tuple",
    "List",
    "Value",
    "Bool",
]

log = logging.getLogger("lab." + __name__)

_dispatch = Dispatcher()


@pytest.fixture(params=[False, True])
def check_lazy_shapes(request):
    if request.param:
        with B.lazy_shapes:
            yield
    else:
        yield


def autograd_box(x):
    """Box a tensor in AutoGrad."""
    t = trace_stack.new_trace().__enter__()
    n = VJPNode.new_root()
    return new_box(x, t, n)


@_dispatch(precedence=1)
def to_np(x: Union[B.NPNumeric, B.Number]):
    """Convert a tensor to NumPy."""
    return x


@_dispatch
def to_np(x: Dimension):
    return unwrap_dimension(x)


@_dispatch
def to_np(x: B.AGNumeric):
    return getval(x)


@_dispatch
def to_np(x: Union[B.TorchNumeric, B.TFNumeric]):
    return x.numpy()


@_dispatch
def to_np(x: B.JAXNumeric):
    return np.array(x)


@_dispatch
def to_np(tup: Union[tuple, tf.TensorShape, torch.Size, Shape]):
    return tuple(to_np(x) for x in tup)


@_dispatch
def to_np(lst: list):
    return to_np(tuple(lst))


@_dispatch
def allclose(x, y, assert_dtype: bool = False, **kw_args):
    """Assert that two numeric objects are close."""
    x, y = to_np(x), to_np(y)

    # Assert that data types are equal if required.
    if assert_dtype:
        assert np.array(x).dtype == np.array(y).dtype

    np.testing.assert_allclose(x, y, **kw_args)


@_dispatch
def allclose(x: tuple, y: tuple, assert_dtype: bool = False, **kw_args):
    assert len(x) == len(y)
    for xi, yi in zip(x, y):
        allclose(xi, yi, assert_dtype=assert_dtype, **kw_args)


def check_function(f, args_spec, kw_args_spec=None, assert_dtype=True, skip=None):
    """Check that a function produces consistent output. Moreover, if the first
    argument is a data type, check that the result is exactly of that type."""
    skip = [] if skip is None else skip

    if kw_args_spec is None:
        kw_args_spec = {}

    # Construct product of keyword arguments.
    kw_args_prod = list(
        product(*[[(k, v) for v in vs.forms()] for k, vs in kw_args_spec.items()])
    )
    kw_args_prod = [{k: v for k, v in kw_args} for kw_args in kw_args_prod]

    # Add default call.
    kw_args_prod += [{}]

    # Construct product of arguments.
    args_prod = list(product(*[arg.forms() for arg in args_spec]))

    # Construct framework types to skip mixes of.
    fw_types = [
        plum.Union(t, plum.List(t), plum.Tuple(t))
        for t in [B.AGNumeric, B.TorchNumeric, B.TFNumeric, B.JAXNumeric]
    ]

    # Construct other types to skip entirely.
    skip_types = [plum.Union(t, plum.List(t), plum.Tuple(t)) for t in skip]

    # Check consistency of results.
    for kw_args in kw_args_prod:
        # Compare everything against the first result.
        first_result = f(*args_prod[0], **kw_args)

        # If first argument is a data type, then check that.
        if isinstance(args_prod[0][0], B.DType):
            assert B.dtype(first_result) is args_prod[0][0]

        for args in args_prod:
            # Skip mixes of FW types.
            fw_count = sum([any(isinstance(arg, t) for arg in args) for t in fw_types])

            # Skip all skips.
            skip_count = sum(
                [any(isinstance(arg, t) for arg in args) for t in skip_types]
            )

            if fw_count >= 2 or skip_count >= 1:
                log.debug(
                    f"Skipping call with arguments {args} and keyword "
                    f"arguments {kw_args}."
                )
                continue

            # Check consistency.
            log.debug(f"Call with arguments {args} and keyword arguments {kw_args}.")
            result = f(*args, **kw_args)
            allclose(first_result, result, assert_dtype=assert_dtype)

            # If first argument is a data type, then again check that.
            if isinstance(args[0], B.DType):
                assert B.dtype(result) is args[0]


class Tensor:
    """Tensor placeholder."""

    def __init__(self, *dims, **kw_args):
        if "mat" not in kw_args or kw_args["mat"] is None:
            self.mat = np.array(np.random.randn(*dims))
        else:
            self.mat = kw_args["mat"]

    def forms(self):
        return [self.np(), self.tf(), self.torch(), self.ag(), self.jax()]

    def np(self):
        return self.mat

    def tf(self):
        return tf.constant(self.mat)

    def torch(self):
        return torch.tensor(self.mat)

    def ag(self):
        return autograd_box(self.mat)

    def jax(self):
        return jnp.array(self.mat)


class PositiveTensor(Tensor):
    """Positive tensor placeholder."""

    def __init__(self, *dims, **kw_args):
        if "mat" not in kw_args or kw_args["mat"] is None:
            mat = np.array(np.random.rand(*dims))
        else:
            mat = kw_args["mat"]
        Tensor.__init__(self, mat=mat)


class BoolTensor(Tensor):
    """Boolean tensor placeholder."""

    def __init__(self, *dims, **kw_args):
        if "mat" not in kw_args or kw_args["mat"] is None:
            mat = np.array(np.random.rand(*dims) > 0.5)
        else:
            mat = kw_args["mat"]
        Tensor.__init__(self, mat=mat)

    def torch(self):
        return torch.tensor(self.mat.astype(np.uint8))


class NaNTensor(Tensor):
    """Tensor containing NaNs placeholder."""

    def __init__(self, *dims, **kw_args):
        if "mat" not in kw_args or kw_args["mat"] is None:
            mat = np.array(np.random.randn(*dims))
            set_nan = np.array(np.random.rand(*dims) > 0.5)
            mat[set_nan] = np.nan
        else:
            mat = kw_args["mat"]
        Tensor.__init__(self, mat=mat)


class Matrix(Tensor):
    """Matrix placeholder."""

    def __init__(self, *shape, **kw_args):
        # Handle shorthands.
        if shape == ():
            shape = (3, 3)
        elif len(shape) == 1:
            shape = shape * 2

        Tensor.__init__(self, *shape, **kw_args)


class PSD(Matrix):
    """Positive-definite tensor placeholder."""

    def __init__(self, *shape):
        # Handle shorthands.
        if shape == ():
            shape = (3, 3)
        elif len(shape) == 1:
            shape = shape * 2

        if not shape[-2] == shape[-1]:
            raise ValueError("PSD matrix must be square.")

        a = np.random.randn(*shape)
        perm = list(range(len(a.shape)))
        perm[-2], perm[-1] = perm[-1], perm[-2]
        a_t = np.transpose(a, perm)
        Matrix.__init__(self, mat=np.matmul(a, a_t))


class PSDTriangular(PSD):
    def __init__(self, *shape, **kw_args):
        PSD.__init__(self, *shape)

        # Zero upper triangular part.
        for i in range(self.mat.shape[0]):
            for j in range(i + 1, self.mat.shape[1]):
                self.mat[..., i, j] = 0

        # Create upper-triangular matrices, if asked for.
        if kw_args.get("upper", False):
            perm = list(range(len(self.mat.shape)))
            perm[-2], perm[-1] = perm[-1], perm[-2]
            self.mat = np.transpose(self.mat, perm)


class Tuple:
    """Tuple placeholder."""

    def __init__(self, *xs):
        self.xs = xs

    def forms(self):
        return map(tuple, zip(*(x.forms() for x in self.xs)))


class List:
    """List placeholder for in argument specification."""

    def __init__(self, *xs):
        self.xs = xs

    def forms(self):
        return map(list, zip(*(x.forms() for x in self.xs)))


class Value:
    """Value placeholder."""

    def __init__(self, *values):
        self._values = values

    def forms(self):
        return self._values


class Bool(Value):
    """Boolean placeholder."""

    def __init__(self):
        Value.__init__(self, False, True)
