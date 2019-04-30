# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from plum import Type

import lab as B
from lab.util import _determine_splitting_index
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, allclose, approx


def test_determine_splitting_index():
    yield eq, _determine_splitting_index((1, 2), -1), 2
    yield eq, _determine_splitting_index((1, 2, 3), -1), 3
    yield eq, _determine_splitting_index((1, 2), 0), 1
    yield eq, _determine_splitting_index((1, 2), 1), 2
    yield eq, _determine_splitting_index((1, 2), None), 0


def test_abstract():
    # Test that the name and docstrings for functions are available.
    yield eq, B.transpose.__name__, 'transpose'
    yield neq, B.transpose.__doc__, ''

    # Test that `abstract` does its job by temporarily modifying the
    # numeric and framework type.
    numeric_types = set(B.Numeric._types)
    B.Numeric._types.add(Type(str))
    framework_types = set(B.Framework._types)
    B.Framework._types.add(Type(str))
    B.dispatch.clear_cache()
    yield raises, NotImplementedError, lambda: B.transpose('')
    B.Numeric._types = numeric_types
    B.Framework._types = framework_types
    B.dispatch.clear_cache()
