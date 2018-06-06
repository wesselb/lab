import unittest
import tensorflow as tf
import numpy as np

from bvn_cdf import bvn_cdf


def shape(x):
    """
    Get the shape of `x`.

    :param x: tensor
    :return: shape of tensor
    """
    return tuple(int(y) for y in x.get_shape())


class TestBvnCdf(unittest.TestCase):
    def setUp(self):
        self.x = tf.constant([.1, .2, .3])
        self.y = tf.constant([.5, .6, .7])
        self.rho = tf.constant([-.33, .33, .66])
        self.out = bvn_cdf(self.x, self.y, self.rho)
        self.ref = np.array([0.3269202833,
                             0.4648044872,
                             0.5571288537])

    def test_correctness(self):
        with tf.Session():
            np.testing.assert_allclose(self.ref, self.out.eval())

    def test_gradient(self):
        with tf.Session():
            inp = [self.x, self.y, self.rho]
            inp_shapes = [shape(self.x), shape(self.y), shape(self.rho)]
            inp_init = [self.x.eval(), self.y.eval(), self.rho.eval()]
            err = tf.test.compute_gradient_error(inp, inp_shapes,
                                                 self.out, shape(self.out),
                                                 inp_init, delta=1e-3)
        np.testing.assert_array_less(err, 1e-3)
