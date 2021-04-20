from functools import wraps

import numpy as np
import plum
import plum.signature
import plum.type

from . import B

__all__ = ["resolve_axis", "as_tuple", "batch_computation", "abstract"]

_dispatch = plum.Dispatcher()


def resolve_axis(a, axis):
    """Resolve axis for a tensor `a`.

    Args:
        a (tensor): Tensor of the axis.
        axis (int): Axis to resolve.

    Return:
        int: Resolved axis.
    """
    if axis < 0:
        return B.rank(a) + axis
    else:
        return axis


@_dispatch
def as_tuple(x: tuple):
    """Get `x` as a tuple. Will be wrapped in a one-tuple if it is not a tuple.

    Args:
        x (object): Object to get as a tuple.

    Returns:
        tuple: `x` as a tuple.
    """
    return x


@_dispatch
def as_tuple(x):
    return (x,)


def _common_shape(*shapes):
    common_shape = shapes[0]
    for shape in shapes[1:]:
        # Add empty dimensions to either shape if it is shorter.
        diff = len(common_shape) - len(shape)
        shape = (1,) * max(diff, 0) + shape
        common_shape = (1,) * max(-diff, 0) + common_shape

        # Resolve the shapes.
        new_common_shape = ()
        for d1, d2 in zip(common_shape, shape):
            if d1 == d2:
                new_common_shape += (d1,)
            elif d1 == 1:
                new_common_shape += (d2,)
            elif d2 == 1:
                new_common_shape += (d1,)
            else:
                raise RuntimeError(
                    f"Cannot reconcile running common shape {common_shape} "
                    f"with {shape}."
                )
        common_shape = new_common_shape
    return common_shape


def _translate_index(index, batch_shape):
    # Remove superfluous index dimensions and cast to tuple.
    index = tuple(index[-len(batch_shape) :])

    # Resolve the index.
    translated_index = ()
    for i, s in zip(index, batch_shape):
        if i < s:
            translated_index += (i,)
        elif s == 1:
            translated_index += (0,)
        else:
            raise RuntimeError(
                f"Cannot translate index {index} to batch shape {batch_shape}."
            )
    return translated_index


def batch_computation(f, xs, ranks):
    """Apply a function over all batches of arguments.

    Args:
        f (function): Function that performs the computation.
        xs (tuple): Matrices or batches of matrices.
        ranks (tuple): Ranks of the arguments.

    Returns:
        tensor: Result in batched form.
    """
    # Reshape arguments for batched computation.
    batch_shapes = [B.shape(x)[:-rank] for x, rank in zip(xs, ranks)]

    # Find the common shape.
    batch_shape = _common_shape(*batch_shapes)
    # Force evaluation of the element of the shape: if the shapes are lazy or when
    # a function is evaluated abstractly, the dimensions of the shape may still be
    # wrapped.
    indices = np.indices(tuple(int(x) for x in batch_shape))

    # Handle the edge case that there is no batching.
    if len(indices) == 0:
        indices = [()]
    else:
        # Put the index dimension last.
        perm = tuple(list(range(1, len(batch_shape) + 1))) + (0,)
        indices = indices.transpose(perm)
        # Turn into a list of indices.
        indices = indices.reshape(-1, len(batch_shape))

    # Loop over batches.
    batches = []
    for index in indices:
        batches.append(
            f(*[x[_translate_index(index, s)] for x, s in zip(xs, batch_shapes)])
        )

    # Construct result, reshape, and return.
    res = B.stack(*batches, axis=0)
    return B.reshape(res, *(batch_shape + B.shape(res)[1:]))


def abstract(promote=-1):
    """Create a decorator for an abstract function.

    Args:
        promote (int, optional): Number of arguments to promote. Set to `-1`
            to promote all arguments, and set to `None` or `0` to promote no
            arguments. Defaults to `-1`.

    Returns:
        function: Decorator.
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kw_args):
            # Determine splitting index.
            if promote is None or promote == 0:
                promote_index = 0
            elif promote < 0:
                promote_index = len(args) + 1
            else:
                promote_index = promote

            # Record types.
            types_before = tuple(plum.type_of(arg) for arg in args)

            # Promote.
            args = plum.promote(*args[:promote_index]) + args[promote_index:]

            # Enforce a change in types. Otherwise, the call will recurse, which
            # means that an implementation is not available.
            types_after = tuple(plum.type_of(arg) for arg in args)
            if types_before == types_after:
                signature = plum.signature.Signature(*types_after)
                # TODO: Use the message from Plum directly here.
                raise plum.NotFoundLookupError(
                    f'For function "{f.__name__}", signature {signature} could not be '
                    f"resolved."
                )

            # Retry call.
            return getattr(B, f.__name__)(*args, **kw_args)

        return wrapper

    return decorator
