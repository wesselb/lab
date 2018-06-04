# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np
import tensorflow as tf
from plum import Dispatcher

from .. import B

_dispatch = Dispatcher()

_TF = {tf.Variable, tf.Tensor}
_Numeric = {Number, np.ndarray} | _TF


@_dispatch(Number)
def dtype(a):
    return np.array(a).dtype


@_dispatch(_Numeric)
def dtype(a):
    return a.dtype


@_dispatch(_Numeric, object)
def cast(a, dtype):
    return tf.cast(a, dtype)


@_dispatch(object)
def cast(a, dtype):
    # This is a call with a keyword argument.
    return cast(a, dtype)


@_dispatch(object)
def shape(a):
    return np.shape(a)


@_dispatch(_TF)
def shape(a):
    return tf.shape(a)


@_dispatch(object)
def shape_int(a):
    return B.shape(a)


@_dispatch(_TF)
def shape_int(a):
    return tuple(int(x) for x in a.get_shape())


def matmul(a, b, tr_a=False, tr_b=False):
    return tf.matmul(a, b, transpose_a=tr_a, transpose_b=tr_b)


def diag(a):
    if B.rank(a) == 1:
        return tf.diag(a)
    elif B.rank(a) == 2:
        return tf.diag_part(a)
    else:
        raise ValueError('Argument must have rank 1 or 2.')


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


dot = matmul

sum = tf.reduce_sum
mean = tf.reduce_mean
prod = tf.reduce_prod
logsumexp = tf.reduce_logsumexp
min = tf.reduce_min
max = tf.reduce_max
all = tf.reduce_all
any = tf.reduce_any

array = tf.constant
eig = tf.self_adjoint_eig
randn = tf.random_normal
rand = tf.random_uniform
