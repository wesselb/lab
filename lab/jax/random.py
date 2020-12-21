import jax
import numpy as np

from . import B, dispatch
from ..types import Int, JAXDType, JAXNumeric

__all__ = []


class JAXRNG:
    """An RNG for JAX."""

    def __init__(self):
        self.set_seed(0)

    def set_seed(self, seed):
        """Set the RNG according to a seed.

        Args:
            seed (int): Seed.
        """
        self.key = jax.random.PRNGKey(seed)

    def split_key(self):
        """Get a key to generate a random number.

        Returns:
            :class:`jax.DeviceArray`: Key.
        """
        self.key, subkey = jax.random.split(self.key)
        return subkey


B.jax_rng = JAXRNG()


@dispatch(JAXDType, [Int])
def rand(dtype, *shape):
    return jax.random.uniform(B.jax_rng.split_key(), shape, dtype=dtype)


@dispatch(JAXDType, [Int])
def randn(dtype, *shape):
    return jax.random.normal(B.jax_rng.split_key(), shape, dtype=dtype)


@dispatch(JAXNumeric, Int)
def choice(a, n):
    inds = np.random.choice(a.shape[0], n, replace=True)
    choices = a[inds]
    return choices[0] if n == 1 else choices
