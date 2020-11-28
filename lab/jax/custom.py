import numpy as np
from jax.core import Primitive
from jax import custom_vjp
from plum import Dispatcher
from functools import wraps

from ..util import as_tuple

__all__ = ['jax_register']
_dispatch = Dispatcher()


def _as_primitive(f):
    def f_wrapped(*args, **kw_args):
        # Convert `args` to NumPy. The implementations do not support Jax types.
        return f(*[np.array(arg) for arg in args], **kw_args)

    primitive = Primitive(f.__name__)
    primitive.def_impl(f_wrapped)

    # Wrap `primitive.bind` to preserve the metadata of `f`.

    @wraps(f)
    def bind_wrapped(*args, **kw_args):
        return primitive.bind(*args, **kw_args)

    return bind_wrapped


def jax_register(f, s_f):
    """Register a function and its sensitivity for Jax.

    Args:
        f (function): Function to register.
        s_f (function): Sensitivity of `f`.

    Returns:
        function: Jax function.
    """
    f = _as_primitive(f)
    s_f = _as_primitive(s_f)

    f = custom_vjp(f)

    # Define and register the forward and backward pass.

    def forward(*args, **kw_args):
        y = f(*args, **kw_args)
        return y, (y, args, kw_args)

    def backward(res, s_y):
        y, args, kw_args = res
        return as_tuple(s_f(s_y, y, *args, **kw_args))

    f.defvjp(forward, backward)

    return f
