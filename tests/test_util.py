# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import plum
import pytest
from plum import NotFoundLookupError

import lab as B
from lab.util import abstract, batch_computation


def test_batch_computation():
    # Correctness is already checked by usage in linear algebra functions. Here
    # we test the check of batch shapes.
    with pytest.raises(ValueError):
        batch_computation(None, B.randn(3, 4, 4), B.randn(2, 4, 4))
    with pytest.raises(ValueError):
        batch_computation(None, B.randn(2, 2, 4, 4), B.randn(2, 4, 4))


def test_metadata():
    # Test that the name and docstrings for functions are available.
    assert B.transpose.__name__ == 'transpose'
    assert B.transpose.__doc__ != ''


def test_abstract():
    class General(object):
        pass

    class Specific(object):
        pass

    a = General()
    b = Specific()

    # Temporarily mock Plum's promotion function.
    plum_promote = plum.promote
    plum.promote = lambda *args: (b,) * len(args)

    # Define some abstract functions.
    @B.dispatch([General])
    @abstract()
    def f1(*args):
        return args

    @B.dispatch([Specific])
    def f1(*args):
        return args

    @B.dispatch([General])
    @abstract(promote=None)
    def f2(*args):
        return args

    @B.dispatch([Specific])
    def f2(*args):
        return args

    @B.dispatch([General])
    @abstract(promote=-1)
    def f3(*args):
        return args

    @B.dispatch([Specific])
    def f3(*args):
        return args

    @B.dispatch([General])
    @abstract(promote=0)
    def f4(*args):
        return args

    @B.dispatch([Specific])
    def f4(*args):
        return args

    @B.dispatch([General])
    @abstract(promote=1)
    def f5(*args):
        return args

    @B.dispatch(Specific, [General])
    def f5(*args):
        return args

    @B.dispatch([General])
    @abstract(promote=2)
    def f6(*args):
        return args

    @B.dispatch(Specific, Specific, [General])
    def f6(*args):
        return args

    # Register methods.
    B.f1 = f1
    B.f2 = f2
    B.f3 = f3
    B.f4 = f4
    B.f5 = f5
    B.f6 = f6

    # Test promotion.
    assert f1(a, a, a) == (b, b, b)
    with pytest.raises(NotFoundLookupError):
        f2(a, a, a)
    assert f3(a, a, a) == (b, b, b)
    with pytest.raises(NotFoundLookupError):
        f4(a, a, a)
    assert f5(a, a, a) == (b, a, a)
    assert f5(a) == (b,)
    assert f6(a, a, a) == (b, b, a)
    assert f6(a, a) == (b, b)

    # Put back promotion function.
    plum.promote = plum_promote
