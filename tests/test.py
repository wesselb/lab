# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, lam
import numpy as np


def test_np_matmul():
    from lab import B
    a, b = np.random.randn(5, 10), np.random.randn(10, 20)
    yield lam, lambda: np.all(np.isclose(B.matmul(a, b), np.matmul(a, b)))
