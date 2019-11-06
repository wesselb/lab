import sys

import numpy as np

from . import dispatch, B
from .types import DType, Int, Numeric
from .util import abstract

__all__ = ['set_random_seed', 'rand', 'randn', 'choice']


@dispatch(Int)
def set_random_seed(seed):
    """Set the random seed for all frameworks.

    Args:
        seed (int): Seed.
    """
    # Set seed in NumPy.
    np.random.seed(seed)

    # Set seed for TensorFlow, if it is loaded.
    if 'tensorflow' in sys.modules:
        import tensorflow
        tensorflow.random.set_seed(seed)

    # Set seed for PyTorch, if it is loaded.
    if 'torch' in sys.modules:
        import torch
        torch.manual_seed(seed)


@dispatch(DType, [Int])
@abstract()
def rand(dtype, *shape):  # pragma: no cover
    """Construct a U[0, 1] random tensor.

    Args:
        dtype (dtype, optional): Data type. Defaults to the default data type.
        *shape (shape, optional): Shape of the tensor. Defaults to `()`.

    Returns:
        tensor: Random tensor.
    """


@dispatch.multi((Int,),  # Single integer is a not a reference.
                ([Int],))
def rand(*shape):
    return rand(B.default_dtype, *shape)


@dispatch(Numeric)
def rand(ref):
    return rand(B.dtype(ref), *B.shape(ref))


@dispatch(DType, [Int])
@abstract(promote=None)
def randn(dtype, *shape):  # pragma: no cover
    """Construct a N(0, 1) random tensor.

    Args:
        dtype (dtype, optional): Data type. Defaults to the default data type.
        *shape (shape, optional): Shape of the tensor. Defaults to `()`.

    Returns:
        tensor: Random tensor.
    """


@dispatch.multi((Int,),  # Single integer is a not a reference.
                ([Int],))
def randn(*shape):
    return randn(B.default_dtype, *shape)


@dispatch(Numeric)
def randn(ref):
    return randn(B.dtype(ref), *B.shape(ref))


@dispatch(Numeric, Int)
@abstract(promote=None)
def choice(a, n):
    """Randomly choose from a tensor *with* replacement.

    Args:
        a (tensor): Tensor to choose from.
        n (int, optional): Number of samples. Defaults to `1`.

    Returns:
        tensor: Samples.
    """


@dispatch(Numeric)
def choice(a):
    return choice(a, 1)
