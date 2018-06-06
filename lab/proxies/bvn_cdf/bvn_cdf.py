# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as tf_dists
from tensorflow.python.framework import ops

__all__ = ['bvn_cdf']

bvn_cdf_op = None


def bvn_cdf(x, y, rho, name=None):
    """
    Standard bivariate normal CDF.

    Args:
        x (vector): First variable.
        y (vector): Second variant.
        rho (vector): Correlation coefficient.
        name (str, optional): Name of operation.
    """
    # Lazily load operation.
    global bvn_cdf_op
    if bvn_cdf_op is None:
        path = os.path.join(os.path.dirname(__file__), 'bvn_cdf_op.so')
        bvn_cdf_op = tf.load_op_library(path).bvn_cdf

    return bvn_cdf_op(x, y, rho, name=name)


@ops.RegisterGradient("BvnCdf")
def _bvn_cdf_grad(op, grad):
    xs, ys, rho = op.inputs[0], op.inputs[1], op.inputs[2]
    q = tf.sqrt(1 - rho ** 2)

    # Compute the densities.
    pdfs = (1 / (2 * np.pi * q) *
            tf.exp(-(xs ** 2 - 2 * rho * xs * ys + ys ** 2) /
                   (2 * (1 - rho ** 2))))

    # Compute sensitivity for correlation coefficient.
    sens_rho = grad * pdfs

    # Compute sensitivity for inputs.
    dist_z = tf_dists.Normal(tf.constant(0., dtype=xs.dtype),
                             tf.constant(1., dtype=xs.dtype))
    dist_x = tf_dists.Normal(rho * xs, q)
    dist_y = tf_dists.Normal(rho * ys, q)

    sens_x = grad * dist_z.prob(xs) * dist_x.cdf(ys)
    sens_y = grad * dist_z.prob(ys) * dist_y.cdf(xs)

    return [sens_x, sens_y, sens_rho]
