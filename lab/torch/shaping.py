# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from . import dispatch, Numeric

__all__ = []


@dispatch(Numeric)
def shape(a):
    s = a.shape
    return tuple(s[i] for i in range(rank(a)))


@dispatch(Numeric)
def shape_int(a):
    return shape(a)


@dispatch(Numeric)
def rank(a):
    return len(a.shape)


@dispatch(Numeric)
def length(a):
    return a.numel()
