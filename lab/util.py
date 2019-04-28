# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from functools import wraps

__all__ = ['abstract']


def abstract(f):
    """Decorator for an abstract function."""

    @wraps(f)
    def wrapper(*args, **kw_args):
        raise NotImplementedError('No implementation of "{}" for argument(s) '
                                  '{}.'.format(f.__name__, args))

    return wrapper
