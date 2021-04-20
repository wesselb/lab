from functools import wraps

import tensorflow as tf
from plum import convert, Dispatcher

from . import B

__all__ = ["tensorflow_register", "as_tf"]

_dispatch = Dispatcher()


@_dispatch
def as_tf(x: B.Numeric):
    """Convert object to TensorFlow.

    Args:
        x (object): Object to convert.

    Returns:
        object: `x` as a TensorFlow object.
    """
    dtype = convert(B.dtype(x), B.TFDType)
    return tf.constant(x, dtype=dtype)


@_dispatch
def as_tf(xs: tuple):
    return tuple([as_tf(x) for x in xs])


def _np_apply(f, out_dtypes, *args, **kw_args):
    """Apply a NumPy function in TensorFlow.

    Args:
        f (function): NumPy function.
        out_dtypes (list[dtype]): List of data types of the output.
        *args (object): Argument to `f`.
        **kw_args (object): Keyword arguments to `f`.

    Returns:
        tensor: Result as a TensorFlow operation.
    """
    return tf.py_function(
        lambda *args_: f(*[arg.numpy() for arg in args_], **kw_args), args, out_dtypes
    )


def tensorflow_register(f, s_f):
    """Register a function and its sensitivity for TensorFlow.

    Args:
        f (function): Function to register.
        s_f (function): Sensitivity of `f`.

    Returns:
        function: TensorFlow primitive.
    """

    @wraps(f)
    def primitive(*args, **kw_args):
        # TODO: This assumes that the output is of the data type of the first input.
        #  Generally, this is *not* true. How to best approach this?
        y = _np_apply(f, args[0].dtype, *args, **kw_args)

        def grad(s_y):
            # TODO: This assumes that the sensitivities of the inputs are of the data
            # types of the inputs. Again, generally, this is *not* true. How to best
            # approach this?
            return _np_apply(
                s_f, [arg.dtype for arg in args], *((s_y, y) + args), **kw_args
            )

        return y, grad

    return tf.custom_gradient(primitive)
