# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from functools import wraps, reduce
from operator import mul

import plum

from . import B

__all__ = ['batch_computation', 'abstract']


def batch_computation(f, *xs):
    """Apply a function over all the batches of the arguments, where the
    arguments are assumed to be matrices or batches of matrices.

    Args:
        *xs (tensor): Matrices or batches of matrices.

    Returns:
        tensor: Result in batched form.
    """
    # Reshape arguments for batched computation.
    batch_shapes = [B.shape(x)[:-2] for x in xs]
    xs = [B.reshape(x, -1, *B.shape(x)[-2:]) for x in xs]

    # Check that all batch shapes are the same.
    if not all(s == batch_shapes[0] for s in batch_shapes[1:]):
        raise ValueError('Inconsistent batch shapes.')

    batch_shape = batch_shapes[0]

    # Loop over batches.
    batches = []
    for i in range(reduce(mul, batch_shape, 1)):
        batches.append(f(*[x[i, :, :] for x in xs]))

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
                signature = plum.Signature(*types_after)
                # TODO: Use the message from Plum directly here.
                raise plum.NotFoundLookupError(
                    'For function "{}", signature {} could not be resolved.'
                    ''.format(f.__name__, signature))

            # Retry call.
            return getattr(B, f.__name__)(*args, **kw_args)

        return wrapper

    return decorator
