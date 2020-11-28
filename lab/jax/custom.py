import numpy as np
from jax import custom_vjp
from plum import Dispatcher

from ..util import as_tuple

__all__ = ['jax_register']
_dispatch = Dispatcher()


def jax_register(f, s_f):
    """Register a function and its sensitivity for Jax.

    Args:
        f (function): Function to register.
        s_f (function): Sensitivity of `f`.

    Returns:
        function: Jax function.
    """
    # Create a primitive for `f`.
    f_primitive = custom_vjp(f)

    def forward(*args, **kw_args):
        # Convert `args` to NumPy. The implementations do not yet support Jax types.
        args = [np.array(arg) for arg in args]
        y = f(*args, **kw_args)
        return y, (y, args, kw_args)

    def backward(res, s_y):
        y, args, kw_args = res
        # Convert `s_y` to NumPy. The implementations do not yet support Jax types.
        s_y = np.array(s_y)
        return as_tuple(s_f(s_y, y, *args, **kw_args))

    # Register the sensitivity.
    f_primitive.defvjp(forward, backward)

    # Return the Jax primitive.
    return f_primitive
