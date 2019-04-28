# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from functools import wraps

from plum import convert, promote

from .types import Framework
from . import B

__all__ = ['abstract']


def abstract(f):
    """Decorator for an abstract function."""

    @wraps(f)
    def wrapper(*args, **kw_args):
        # Record types.
        types_before = tuple(type(arg) for arg in args)

        # Convert and promote.
        args = promote(*[convert(arg, Framework) for arg in args])

        # Enforce a change in types. Otherwise, the call will recurse.
        types_after = tuple(type(arg) for arg in args)
        if types_before == types_after:
            raise NotImplementedError(
                'No implementation of "{}" for argument(s) {}.'
                ''.format(f.__name__, args))

        # Retry call.
        return getattr(B, f.__name__)(*args, **kw_args)

    return wrapper
