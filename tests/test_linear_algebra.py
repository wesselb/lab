# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import lab as B
from . import check_function, Matrix, Bool
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, lam, allclose, approx


def test_matmul():
    yield check_function, \
          B.matmul, \
          (Matrix(), Matrix()), \
          {'tr_a': Bool(), 'tr_b': Bool()}
