import sys

import numpy as np
from plum.type import VarArgs

from . import dispatch, B
from .types import DType, Int, Numeric
from .util import abstract

__all__ = ["set_random_seed", "rand", "randn", "choice"]


@dispatch
def set_random_seed(seed: Int):
    """Set the random seed for all frameworks.

    Args:
        seed (int): Seed.
    """
    # Set seed in NumPy.
    np.random.seed(seed)

    # Set seed for TensorFlow, if it is loaded.
    if "tensorflow" in sys.modules:
        import tensorflow

        tensorflow.random.set_seed(seed)

    # Set seed for PyTorch, if it is loaded.
    if "torch" in sys.modules:
        import torch

        torch.manual_seed(seed)

    # Set seed for JAX, if it is loaded.
    if "jax" in sys.modules:
        B.jax_rng.set_seed(seed)


@dispatch
@abstract()
def rand(dtype: DType, *shape: Int):  # pragma: no cover
    """Construct a U[0, 1] random tensor.

    Args:
        dtype (dtype, optional): Data type. Defaults to the default data type.
        *shape (shape, optional): Shape of the tensor. Defaults to `()`.

    Returns:
        tensor: Random tensor.
    """


@dispatch.multi((Int,), (VarArgs(Int),))  # Single integer is a not a reference.
def rand(*shape: Int):
    return rand(B.default_dtype, *shape)


@dispatch
def rand(ref: Numeric):
    return rand(B.dtype(ref), *B.shape(ref))


@dispatch
@abstract(promote=None)
def randn(dtype: DType, *shape: Int):  # pragma: no cover
    """Construct a N(0, 1) random tensor.

    Args:
        dtype (dtype, optional): Data type. Defaults to the default data type.
        *shape (shape, optional): Shape of the tensor. Defaults to `()`.

    Returns:
        tensor: Random tensor.
    """


@dispatch.multi((Int,), (VarArgs(Int),))  # Single integer is a not a reference.
def randn(*shape: Int):
    return randn(B.default_dtype, *shape)


@dispatch
def randn(ref: Numeric):
    return randn(B.dtype(ref), *B.shape(ref))


@dispatch
@abstract(promote=None)
def choice(a: Numeric, n: Int):
    """Randomly choose from a tensor *with* replacement.

    Args:
        a (tensor): Tensor to choose from.
        n (int, optional): Number of samples. Defaults to `1`.

    Returns:
        tensor: Samples.
    """


@dispatch
def choice(a: Numeric):
    return choice(a, 1)
