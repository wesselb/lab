# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np
import tensorflow as tf
from plum import Dispatcher

# noinspection PyUnresolvedReferences
from .bvn_cdf import bvn_cdf
from .. import B

_dispatch = Dispatcher()

_TF = {tf.Variable, tf.Tensor}
_Numeric = {Number, np.ndarray} | _TF


@_dispatch(Number)
def dtype(a):
    return np.array(a).dtype


@_dispatch(np.ndarray)
def dtype(a):
    return a.dtype


@_dispatch(_TF)
def dtype(a):
    return a.dtype.as_numpy_dtype


@_dispatch(_Numeric, [object])
def cast(a, dtype):
    return tf.cast(a, dtype)


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


def take(a, indices, axis=0):
    if axis == 0:
        return tf.gather(a, indices)
    else:
        # Create a permutation to switch `axis` and `0`.
        perm = [i for i in range(B.rank(a))]
        perm[axis], perm[0] = 0, axis

        # Perform gathering.
        return tf.transpose(tf.gather(tf.transpose(a, perm), indices), perm)


def svd(a, full_matrices=False, compute_uv=True, name=None):
    res = tf.svd(a,
                 full_matrices=full_matrices,
                 compute_uv=compute_uv,
                 name=name)
    return res[1], res[0], res[2] if compute_uv else res


def vec_to_tril(a):
    if B.rank(a) != 1:
        raise ValueError('Input must be rank 1.')

    n = B.shape_int(a)[0]
    m = int(((1 + 8 * n) ** .5 - 1) / 2)
    return tf.scatter_nd(indices=list(zip(*np.tril_indices(m))),
                         shape=[m, m],
                         updates=a)


def tril_to_vec(a):
    if B.rank(a) != 2:
        raise ValueError('Input must be rank 2.')
    n, m = B.shape_int(a)
    if n != m:
        raise ValueError('input must be square')
    return tf.gather_nd(a, list(zip(*np.tril_indices(n))))


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

power = tf.pow

# Neural net activations:
sigmoid = tf.nn.sigmoid
tanh = tf.nn.tanh
relu = tf.nn.relu
leaky_relu = tf.nn.leaky_relu
