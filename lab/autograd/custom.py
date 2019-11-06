from autograd.extend import primitive, defvjp_argnums
from plum import Dispatcher

__all__ = ['autograd_register']
_dispatch = Dispatcher()


@_dispatch(tuple)
def as_tuple(x):
    """Get `x` as a tuple. Will be wrapped in a one-tuple if it is not a tuple.

    Args:
        x (object): Object to get as a tuple.

    Returns:
        tuple: `x` as a tuple.
    """
    return x


@_dispatch(object)
def as_tuple(x):
    return (x,)


def autograd_register(f, s_f):
    """Register a function and its sensitivity for AutoGrad.

    Args:
        f (function): Function to register.
        s_f (function): Sensitivity of `f`.

    Returns:
        function: AutoGrad primitive.
    """
    # Create a primitive for `f`.
    f_primitive = primitive(f)

    # Register the sensitivity.
    def vjp_argnums(nums, y, args, kw_args):
        def vjp(s_y):
            grads = as_tuple(s_f(s_y, y, *args, **kw_args))
            return tuple([grads[i] for i in nums])

        return vjp

    defvjp_argnums(f_primitive, vjp_argnums)

    # Return the AutoGrad primitive.
    return f_primitive
