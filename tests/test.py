# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from lab import B

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, lam, ok
import numpy as np
from numpy.testing import assert_allclose


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


def test_length():
    B.backend_to_np()
    yield eq, B.length(np.ones((10, 20, 30))), 10 * 20 * 30


def test_take():
    B.backend_to_tf()
    s = tf.Session()
    a = np.random.randn(10, 50)
    inds = (1, 2, 5, 8)
    yield assert_allclose, np.take(a, inds, 0), s.run(B.take(a, inds, 0))
    yield assert_allclose, np.take(a, inds, 1), s.run(B.take(a, inds, 1))


def test_vec_to_tril_and_back():
    A = np.tril(np.random.randn(10, 10))

    # Test NumPy implementation.
    B.backend_to_np()
    vec_np = B.tril_to_vec(A)
    A_np = B.vec_to_tril(vec_np)
    yield assert_allclose, A_np, A

    # Test TensorFlow implementation.
    B.backend_to_tf()
    s = tf.Session()
    vec_tf = s.run(B.tril_to_vec(A))
    A_tf = s.run(B.vec_to_tril(vec_tf))
    yield assert_allclose, A_tf, A

    # Compare NumPy and TensorFlow implementations.
    yield assert_allclose, A_np, A_tf


def test_promotion():
    B.backend_to_np()

    yield eq, B.promote(1, 1), (1, 1)
    yield eq, B.promote(1., 1.), (1., 1.)
    yield raises, RuntimeError, lambda: B.promote(1, 1.)
    yield raises, RuntimeError, lambda: B.promote(1., 1)

    B.add_promotion_rule(int, float, float)

    yield raises, RuntimeError, lambda: B.promote(1, 1.)
    yield raises, RuntimeError, lambda: B.promote(1., 1)

    B.convert.extend(int, float)(lambda x, _: float(x))

    yield eq, B.promote(1, 1.), (1., 1.)
    yield eq, B.promote(1., 1), (1., 1.)

    yield raises, RuntimeError, lambda: B.promote(1, '1')
    yield raises, RuntimeError, lambda: B.promote('1', 1)
    yield raises, RuntimeError, lambda: B.promote(1., '1')
    yield raises, RuntimeError, lambda: B.promote('1', 1.)

    B.add_promotion_rule(str, {int, float}, {int, float})
    B.convert.extend(str, {int, float})(lambda x, _: float(x))

    yield eq, B.promote(1, '1'), (1., 1.)
    yield eq, B.promote('1', 1), (1., 1.)
    yield eq, B.promote(1., '1'), (1., 1.)
    yield eq, B.promote('1', 1.), (1., 1.)

    B.add_promotion_rule(str, int, float)
    B.add_promotion_rule(str, float, float)
    B.convert.extend(str, float)(lambda x, _: 'lel')

    yield eq, B.promote(1, '1'), (1., 'lel')
    yield eq, B.promote('1', 1), ('lel', 1.)
    yield eq, B.promote(1., '1'), (1., 'lel')
    yield eq, B.promote('1', 1.), ('lel', 1.)
