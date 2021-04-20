from functools import wraps

import jax.numpy as jnp
from jax import custom_vjp
from jax.core import Primitive
from plum import Dispatcher

from . import B
from ..util import as_tuple

__all__ = ["jax_register"]

_dispatch = Dispatcher()


@_dispatch
def as_jax(x: B.Numeric):
    """Convert object to JAX.

    Args:
        x (object): Object to convert.

    Returns:
        object: `x` as a JAX object.
    """
    return jnp.asarray(x)


@_dispatch
def as_jax(xs: tuple):
    return tuple([as_jax(x) for x in xs])


def _as_primitive(f):
    def f_wrapped(*args, **kw_args):
        return as_jax(f(*B.to_numpy(args), **kw_args))

    primitive = Primitive(f.__name__)
    primitive.def_impl(f_wrapped)

    # Wrap `primitive.bind` to preserve the metadata of `f`.

    @wraps(f)
    def bind_wrapped(*args, **kw_args):
        return primitive.bind(*args, **kw_args)

    return bind_wrapped


def jax_register(f, s_f):
    """Register a function and its sensitivity for JAX.

    Args:
        f (function): Function to register.
        s_f (function): Sensitivity of `f`.

    Returns:
        function: JAX function.
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
