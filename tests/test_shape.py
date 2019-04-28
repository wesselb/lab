# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import lab as B
from . import check_function, Tensor
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, lam, allclose, approx


def test_sizing():
    for f in [B.shape, B.shape_int, B.rank, B.length]:
        yield check_function, f, (Tensor(),), {}
        yield check_function, f, (Tensor(3, ),), {}
        yield check_function, f, (Tensor(3, 4),), {}
        yield check_function, f, (Tensor(3, 4, 5),), {}
