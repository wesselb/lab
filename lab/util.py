# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from functools import wraps

import plum

from . import B

__all__ = ['abstract']


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
                signature = plum.Tuple(*types_after)
                # TODO: Use the message from Plum directly here.
                raise plum.NotFoundLookupError(
                    'For function "{}", signature {} could not be resolved.'
                    ''.format(f.__name__, signature))

            # Retry call.
            return getattr(B, f.__name__)(*args, **kw_args)

        return wrapper

    return decorator
