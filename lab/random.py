import sys
from functools import reduce
from operator import mul

import numpy as np
from plum.type import VarArgs, Union

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
    "randcat",
    "choice",
    "randint",
    "randperm",
    "randgamma",
    "randbeta",
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
        for non-LAB random calls, like `tf.random.normal`. Use `B.set_random_seed`
        instead!

    Args:
        state (random state): Random state to set.
    """


@dispatch
def global_random_state(a):
    return global_random_state(B.dtype(a))


@dispatch
@abstract()
def rand(state: RandomState, dtype: DType, *shape: Int):  # pragma: no cover
    """Construct a U[0, 1] random tensor.

    Args:
        state (random state, optional): Random state.
        dtype (dtype, optional): Data type. Defaults to the default data type.
        *shape (shape, optional): Shape of the sample. Defaults to `()`.

    Returns:
        state (random state, optional): Random state.
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
        *shape (shape, optional): Shape of the sample. Defaults to `()`.

    Returns:
        state (random state, optional): Random state.
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
def randcat(state: RandomState, p: Union[Numeric, None], *shape: Int):
    """Randomly draw from a categorical random variable.

    Args:
        state (random state, optional): Random state.
        p (tensor): Probabilities. The last axis determines the probabilities and
            any prior axes add to the shap of the sample.
        *shape (int): Shape of the sample. Defaults to `()`.

    Returns:
        state (random state, optional): Random state.
        tensor: Realisation.
    """
    n = reduce(mul, shape, 1)
    state, sample = randcat(state, p, n)
    return state, B.reshape(sample, *shape, *B.shape(sample)[1:])


def _randcat_last_first(a):
    """Put the last dimension first.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: `a`, but with last dimension first.
    """
    perm = list(range(B.rank(a)))
    return B.transpose(a, perm=perm[-1:] + perm[:-1])


@dispatch
def choice(
    state: RandomState,
    a: Numeric,
    *shape: Int,
    p: Union[Numeric, None] = None,
):
    """Randomly choose from a tensor *with* replacement.

    Args:
        state (random state, optional): Random state.
        a (tensor): Tensor to choose from. Choices will be made along the first
            dimension.
        *shape (int): Shape of the sample. Defaults to `()`.
        p (tensor, optional): Probabilities to sample with.

    Returns:
        state (random state, optional): Random state.
        tensor: Choices.
    """
    if p is None:
        with B.on_device(a):
            p = B.ones(B.dtype_float(a), B.shape(a, 0))
    state, inds = B.randcat(state, p, *shape)
    choices = B.reshape(
        B.take(a, B.flatten(inds), axis=0),
        *B.shape(inds),
        *B.shape(a)[1:],
    )
    return state, choices


@dispatch
def choice(
    a: Numeric,
    *shape: Int,
    p: Union[Numeric, None] = None,
):
    state = B.global_random_state(a)
    state, choices = choice(state, a, *shape, p=p)
    B.set_global_random_state(state)
    return choices


@dispatch
@abstract()
def randint(
    state: RandomState,
    dtype: DType,
    *shape: Int,
    lower: Int = 0,
    upper: Int,
):  # pragma: no cover
    """Construct a tensor of random integers in [`lower`, `upper`).

    Args:
        state (random state, optional): Random state.
        dtype (dtype, optional): Data type. Defaults to the default data type.
        *shape (shape, optional): Shape of the tensor. Defaults to `()`.
        lower (int, optional): Lower bound. Defaults to `0`.
        upper (int): Upper bound. Must be given as a keyword argument.

    Returns:
        state (random state, optional): Random state.
        tensor: Random tensor.
    """


@dispatch.multi((Int,), (VarArgs(Int),))  # Single integer is a not a reference.
def randint(*shape: Int, lower: Int = 0, upper: Int):
    return randint(B.default_dtype, *shape, lower=lower, upper=upper)


@dispatch
def randint(state: RandomState, ref: Numeric, *, lower: Int = 0, upper: Int):
    return randint(state, B.dtype(ref), *B.shape(ref), lower=lower, upper=upper)


@dispatch
def randint(ref: Numeric, *, lower: Int = 0, upper: Int):
    return randint(B.dtype(ref), *B.shape(ref), lower=lower, upper=upper)


@dispatch
@abstract()
def randperm(state: RandomState, dtype: DType, n: Int):  # pragma: no cover
    """Construct a random permutation counting to `n`.

    Args:
        state (random state, optional): Random state.
        dtype (dtype, optional): Data type. Defaults to the default data type.
        n (int): Length of the permutation.

    Returns:
        state (random state, optional): Random state.
        tensor: Random permutation.
    """


@dispatch
def randperm(n: Int):
    return randperm(B.default_dtype, n)


@dispatch
@abstract()
def randgamma(
    state: RandomState,
    dtype: DType,
    *shape: Int,
    alpha: Numeric,
    scale: Numeric,
):  # pragma: no cover
    """Construct a tensor of gamma random variables with shape parameter `alpha` and
        scale `scale`.

    Args:
        state (random state, optional): Random state.
        dtype (dtype, optional): Data type. Defaults to the default data type.
        *shape (shape, optional): Shape of the tensor. Defaults to `()`.
        alpha (scalar): Shape parameter.
        scale (scalar): Scale parameter.

    Returns:
        state (random state, optional): Random state.
        tensor: Random tensor.
    """


@dispatch.multi((Int,), (VarArgs(Int),))  # Single integer is a not a reference.
def randgamma(*shape: Int, alpha: Numeric, scale: Numeric):
    return randgamma(B.default_dtype, *shape, alpha=alpha, scale=scale)


@dispatch
def randgamma(state: RandomState, ref: Numeric, *, alpha: Numeric, scale: Numeric):
    return randgamma(state, B.dtype(ref), *B.shape(ref), alpha=alpha, scale=scale)


@dispatch
def randgamma(ref: Numeric, *, alpha: Numeric, scale: Numeric):
    return randgamma(B.dtype(ref), *B.shape(ref), alpha=alpha, scale=scale)


@dispatch
def randbeta(
    state: RandomState,
    dtype: DType,
    *shape: Int,
    alpha: Numeric,
    beta: Numeric,
):
    """Construct a tensor of beta random variables with shape parameters `alpha` and
        `beta`.

    Args:
        state (random state, optional): Random state.
        dtype (dtype, optional): Data type. Defaults to the default data type.
        *shape (shape, optional): Shape of the tensor. Defaults to `()`.
        alpha (scalar): Shape parameter `alpha`.
        beta (scalar): Shape parameter `beta`.

    Returns:
        state (random state, optional): Random state.
        tensor: Random tensor.
    """
    state, x = randgamma(state, dtype, *shape, alpha=alpha, scale=1)
    state, y = randgamma(state, dtype, *shape, alpha=beta, scale=1)
    return state, x / (x + y)


@dispatch
def randbeta(dtype: DType, *shape: Int, alpha: Numeric, beta: Numeric):
    return randbeta(
        B.global_random_state(dtype),
        dtype,
        *shape,
        alpha=alpha,
        beta=beta,
    )[1]


@dispatch.multi((Int,), (VarArgs(Int),))  # Single integer is a not a reference.
def randbeta(*shape: Int, alpha: Numeric, beta: Numeric):
    return randbeta(B.default_dtype, *shape, alpha=alpha, beta=beta)


@dispatch
def randbeta(state: RandomState, ref: Numeric, *, alpha: Numeric, beta: Numeric):
    return randbeta(state, B.dtype(ref), *B.shape(ref), alpha=alpha, beta=beta)


@dispatch
def randbeta(ref: Numeric, *, alpha: Numeric, beta: Numeric):
    return randbeta(B.dtype(ref), *B.shape(ref), alpha=alpha, beta=beta)
