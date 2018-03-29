# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

from .. import B
from ..util import Namespace

__all__ = ['array', 'shape', 'matmul', 'diag', 'eye', 'randn', 'zeros',
           'trisolve', 'kron', 'concatenate', 'log', 'linalg', 'cholesky',
           'minimum', 'concat', 'cast', 'conj', 'cos', 'dot', 'exp', 'maximum',
           'random', 'sin', 'sum']


def _default32(dtype):
    return tf.float32 if dtype is None else dtype


def array(a, dtype=None):
    if isinstance(a, tf.Tensor) or isinstance(a, tf.Variable):
        if dtype is not None and dtype is not a.dtype:
            return B.cast(a, dtype=dtype)
        else:
            return a
    else:
        return tf.constant(a, dtype=dtype)


def shape(a):
    if isinstance(a, tf.Tensor) or isinstance(a, tf.Variable):
        return tuple(int(x) for x in a.get_shape())
    else:
        return np.shape(a)


def matmul(a, b, tr_a=False, tr_b=False):
    return tf.matmul(a, b, transpose_a=tr_a, transpose_b=tr_b)


def diag(a):
    if B.rank(a) == 1:
        return tf.diag(a)
    elif B.rank(a) == 2:
        return tf.diag_part(a)
    else:
        raise ValueError('Argument must have rank 1 or 2.')


def eye(n, M=None, dtype=None):
    return tf.eye(n, num_columns=M, dtype=_default32(dtype))


def randn(shape, dtype=None):
    return tf.random_normal(shape, dtype=_default32(dtype))


def zeros(shape, dtype=None):
    return tf.zeros(shape, dtype=_default32(dtype))


def trisolve(a, b, tr_a=False, lower=True):
    if tr_a:
        a = tf.conj(a)
    return tf.matrix_triangular_solve(a, b, adjoint=tr_a, lower=lower)


def kron(a, b):
    a_shape, b_shape = B.shape(a), B.shape(b)
    if B.rank(a) != B.rank(b):
        raise ValueError('Inputs must have equal rank.')
    if a_shape[:-2] != b_shape[:-2]:
        raise ValueError('Most-inner parts of the inputs must have equal '
                         'dimensionalities.')
    return tf.reshape(a[..., :, None, :, None] *
                      b[..., None, :, None, :],
                      a_shape[:-2] + (a_shape[-2] * b_shape[-2],
                                      a_shape[-1] * b_shape[-1]))

def cast(a, dtype=None):
    return a if dtype is None else tf.cast(a, dtype)


cholesky = tf.cholesky
cholesky_solve = tf.cholesky_solve
linalg = Namespace()
linalg.cholesky = cholesky

dot = matmul
sum = tf.reduce_sum
concatenate = tf.concat
concat = concatenate
transpose = tf.transpose
sign = tf.sign
trace = tf.trace
eig = tf.self_adjoint_eig
abs = tf.abs

conj = tf.conj
log = tf.log
exp = tf.exp
sin = tf.sin
cos = tf.cos

random = Namespace()
# Call through proxy to coop with changed defaults.
random.randn = lambda *args: B.randn(args)

ones = tf.ones
minimum = tf.minimum
maximum = tf.maximum
