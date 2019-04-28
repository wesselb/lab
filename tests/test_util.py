# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import lab as B
from plum import Type
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, lam, allclose, approx


def test_abstract():
    # Test that the name and docstrings for functions are available.
    yield eq, B.transpose.__name__, 'transpose'
    yield neq, B.transpose.__doc__, ''

    # Test that `abstract` does its job by temporarily modifying the numeric
    # type.
    B.Numeric._types += (Type(str),)
    B.dispatch.clear_cache()
    yield raises, NotImplementedError, lambda: B.transpose('')
    B.Numeric._types = B.Numeric._types[:-1]
    B.dispatch.clear_cache()
