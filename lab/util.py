# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from functools import wraps

from plum import convert, promote

from .types import Framework
from . import B

__all__ = ['abstract']


def abstract(convert_to=-1, promote_to=-1):
    """Create a decorator for an abstract function.

    Args:
        convert_to (int, optional): Index to which to convert arguments. Set to
            `None` to convert no arguments. Defaults to `-1`.
        promote_to (int, optional): Index to which to promote arguments. Set to
            `None` to promote no arguments. Defaults to `-1`.

    Returns:
        function: Decorator.
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kw_args):
            # Fix negative indices: adding `1` below will otherwise break.
            if convert_to is None:
                convert_index = 0
            elif convert_to < 0:
                convert_index = convert_to + len(args) + 1
            else:
                convert_index = convert_to + 1

            if promote_to is None:
                promote_index = 0
            elif promote_to < 0:
                promote_index = promote_to + len(args) + 1
            else:
                promote_index = promote_to + 1

            # Record types.
            types_before = tuple(type(arg) for arg in args)

            # Convert and promote.
            args = tuple(convert(x, Framework)
                         for x in args[:convert_index]) + args[convert_index:]
            args = promote(*args[:promote_index]) + args[promote_index:]

            # Enforce a change in types. Otherwise, the call will recurse.
            types_after = tuple(type(arg) for arg in args)
            if types_before == types_after:
                raise NotImplementedError(
                    'No implementation of "{}" for argument(s) {}.'
                    ''.format(f.__name__, args))

            # Retry call.
            return getattr(B, f.__name__)(*args, **kw_args)

        return wrapper

    return decorator
