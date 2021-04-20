import jax

from . import B, dispatch
from ..types import Int, JAXDType, JAXNumeric

__all__ = []


class JAXRNG:
    """An RNG for JAX.

    Attributes:
        key (:class:`jax.DeviceArray`): Current key.
    """

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


@dispatch
def rand(dtype: JAXDType, *shape: Int):
    return B.move_to_active_device(
        jax.random.uniform(B.jax_rng.split_key(), shape, dtype=dtype)
    )


@dispatch
def randn(dtype: JAXDType, *shape: Int):
    return B.move_to_active_device(
        jax.random.normal(B.jax_rng.split_key(), shape, dtype=dtype)
    )


@dispatch
def choice(a: JAXNumeric, n: Int):
    inds = jax.random.choice(B.jax_rng.split_key(), a.shape[0], (n,), replace=True)
    choices = a[inds]
    return choices[0] if n == 1 else choices
