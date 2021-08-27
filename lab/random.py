import sys

import numpy as np
from plum.type import VarArgs

from . import dispatch, B
from .types import DType, Int, Numeric, RandomState
from .util import abstract

__all__ = ["set_random_seed", "create_random_state", "rand", "randn", "choice"]


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
    if hasattr(B, "jax_global_randomstate"):
        import jax

        B.jax_global_randomstate = jax.random.PRNGKey(seed=seed)


@dispatch
@abstract()
def create_random_state(dtype: DType, seed: Int = 0):
    """Create a random state.

    Args:
        dtype (dtype): Data type of the desired framework to create a random state
            for.
        seed (int, optional): Seed to initialise the random state with. Defaults
            to `0`.

    Returns:
        random state: Random state.
    """


@dispatch
@abstract()
def rand(dtype: DType, *shape: Int):  # pragma: no cover
    """Construct a U[0, 1] random tensor.

    Args:
        dtype (dtype, optional): Data type. Defaults to the default data type.
        *shape (shape, optional): Shape of the tensor. Defaults to `()`.

    Returns:
        tensor or tuple[random state, tensor]: Random tensor if no random state was
            given or a tuple containing the updated random state and the random tensor
            otherwise.
    """


@dispatch.multi((Int,), (VarArgs(Int),))  # Single integer is a not a reference.
def rand(*shape: Int):
    return rand(B.default_dtype, *shape)


@dispatch
def rand(state: RandomState, ref: Numeric):
    return rand(state, B.dtype(ref), *B.shape(ref))


@dispatch
def rand(ref: Numeric):
    return rand(B.dtype(ref), *B.shape(ref))


@dispatch
@abstract()
def randn(state: RandomState, dtype: DType, *shape: Int):  # pragma: no cover
    """Construct a N(0, 1) random tensor.

    Args:
        state (random state, optional): Random state.
        dtype (dtype, optional): Data type. Defaults to the default data type.
        *shape (shape, optional): Shape of the tensor. Defaults to `()`.

    Returns:
        tensor or tuple[random state, tensor]: Random tensor if no random state was
            given or a tuple containing the updated random state and the random tensor
            otherwise.
    """


@dispatch.multi((Int,), (VarArgs(Int),))  # Single integer is a not a reference.
def randn(*shape: Int):
    return randn(B.default_dtype, *shape)


@dispatch
def randn(state: RandomState, ref: Numeric):
    return randn(state, B.dtype(ref), *B.shape(ref))


@dispatch
def randn(ref: Numeric):
    return randn(B.dtype(ref), *B.shape(ref))


@dispatch
@abstract()
def choice(state: RandomState, a: Numeric, n: Int):  # pragma: no cover
    """Randomly choose from a tensor *with* replacement.

    Args:
        state (random state, optional): Random state.
        a (tensor): Tensor to choose from.
        n (int, optional): Number of samples. Defaults to `1`.

    Returns:
        tensor or tuple[random state, tensor]: Samples if no random state was given or
            a tuple containing the updated random state and the samples otherwise.
    """


@dispatch
def choice(state: RandomState, a: Numeric):
    return choice(state, a, 1)


@dispatch
def choice(a: Numeric):
    return choice(a, 1)
