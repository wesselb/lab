from functools import wraps

from jax import ShapeDtypeStruct, custom_vjp, pure_callback
from plum import Dispatcher, convert

from ..custom import TensorDescription

__all__ = ["jax_register"]

_dispatch = Dispatcher()


@_dispatch
def parse_inference_result(x: TensorDescription):
    """Parse the result inference functions to a PyTree (JAX terminology) of
    :class:`jax.ShapeDtypeStruct`s.

    Args:
        x (PyTree): Input to parse.

    Returns:
        PyTree: Parsed input.
    """
    return ShapeDtypeStruct(x.shape, x.dtype)


@_dispatch
def parse_inference_result(xs: tuple):
    return tuple(parse_inference_result(x) for x in xs)


def _wrap_cb(f, i_f):
    @wraps(f)
    def f_wrapped(*args, **kw_args):
        return pure_callback(
            f,
            parse_inference_result(i_f(*args, **kw_args)),
            *args,
            **kw_args,
        )

    return f_wrapped


def jax_register(f, i_f, s_f, i_s_f):
    """Register a function and its sensitivity for JAX.

    Args:
        f (function): Function to register.
        i_f (function): Function that infers the shape of the output.
        s_f (function): Sensitivity of `f`.
        i_s_f (function): Function that infers the shape of the output of the
            sensitivity of `f`.

    Returns:
        function: JAX function.
    """
    f = _wrap_cb(f, i_f)
    s_f = _wrap_cb(s_f, i_s_f)

    f = custom_vjp(f)

    # Define and register the forward and backward pass.

    def forward(*args, **kw_args):
        y = f(*args, **kw_args)
        return y, (y, args, kw_args)

    def backward(res, s_y):
        y, args, kw_args = res
        return convert(s_f(s_y, y, *args, **kw_args), tuple)

    f.defvjp(forward, backward)

    return f
