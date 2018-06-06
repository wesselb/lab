# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from lab import B

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, lam, ok
import numpy as np


def test_np_matmul():
    B.backend_to_np()
    a, b = np.random.randn(5, 10), np.random.randn(10, 20)
    yield lam, lambda: np.all(np.isclose(B.matmul(a, b), np.matmul(a, b)))


def tf_bvn_cdf():
    # This test cannot run on CI, unfortunately.

    B.backend_to_tf()
    with tf.Session():
        for dtype in [np.float32, np.float64]:
            # Test computation.
            x = B.constant([.1, .2, .3], dtype)
            y = B.constant([.5, .6, .7], dtype)
            rho = B.constant([-.33, .33, .66], dtype)
            out = B.bvn_cdf(x, y, rho)
            yield ok, np.allclose(out.eval(), [0.3269202833,
                                               0.4648044872,
                                               0.5571288537])

            # Test gradient.
            inp = [x, y, rho]
            inp_shapes = [B.shape_int(x), B.shape_int(y), B.shape_int(rho)]
            inp_init = [x.eval(), y.eval(), rho.eval()]
            err = tf.test.compute_gradient_error(
                inp, inp_shapes, out, B.shape_int(out), inp_init)
            yield le, err, 1e-4
