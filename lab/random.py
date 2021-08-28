import sys

import numpy as np
from plum.type import VarArgs

from . import dispatch, B
from .types import DType, Int, Numeric, RandomState
from .util import abstract

__all__ = [
    "set_random_seed",
    "create_random_state",
    "global_random_state",
    "set_global_random_state",
    "rand",
    "randn",
    "choice",
]


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
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.random.set_global_generator(tf.random.Generator.from_seed(seed))

    # Set seed for PyTorch, if it is loaded.
    if "torch" in sys.modules:
        import torch

        torch.manual_seed(seed)

    # Set seed for JAX, if it is loaded.
    if hasattr(B, "jax_global_random_state"):
        import jax

        B.jax_global_random_state = jax.random.PRNGKey(seed=seed)


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
def global_random_state(dtype: DType):
    """Get the global random state.

    Args:
        dtype (dtype): Data type of the desired framework for which to get the global
            random state.

    Returns:
        random state: Global random state.
    """


@dispatch
@abstract()
def set_global_random_state(state: RandomState):
    """Set the global random state.

    NOTE:
        In TensorFlow, setting the global random state does NOT fix the randomness
        for non-LAB random calls, like `tf.random.normal`. Use `B.set_seed` instead!

    Args:
        state (random state): Random state to set.
    """


@dispatch
def global_random_state(a):
    return global_random_state(B.dtype(a))


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
        tensor: Random tensor.
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
        tensor: Choices.
    """


@dispatch
def choice(state: RandomState, a: Numeric):
    return choice(state, a, 1)


@dispatch
def choice(a: Numeric):
    return choice(a, 1)
