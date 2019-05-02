# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import plum
from plum import NotFoundLookupError

import lab as B
from lab.util import abstract
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, allclose, approx


def test_metadata():
    # Test that the name and docstrings for functions are available.
    yield eq, B.transpose.__name__, 'transpose'
    yield neq, B.transpose.__doc__, ''


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
    yield eq, f1(a, a, a), (b, b, b)
    yield raises, NotFoundLookupError, lambda: f2(a, a, a)
    yield eq, f3(a, a, a), (b, b, b)
    yield raises, NotFoundLookupError, lambda: f4(a, a, a)
    yield eq, f5(a, a, a), (b, a, a)
    yield eq, f5(a), (b,)
    yield eq, f6(a, a, a), (b, b, a)
    yield eq, f6(a, a), (b, b)

    # Put back promotion function.
    plum.promote = plum_promote
