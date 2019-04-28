# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from . import dispatch, TF, B

__all__ = []


@dispatch(TF, TF)
def matmul(a, b, tr_a=False, tr_b=False):
    return tf.matmul(a, b, transpose_a=tr_a, transpose_b=tr_b)


@dispatch(TF)
def transpose(a):
    return tf.transpose(a)


@dispatch(TF)
def trace(a, axis1=0, axis2=1):
    perm = [i for i in range(B.rank(a)) if i not in [axis1, axis2]]
    perm += [axis1, axis2]
    a = tf.transpose(a, perm=perm)
    return tf.trace(a)


@dispatch(TF, TF)
def kron(a, b):
    shape_a = B.shape_int(a)
    shape_b = B.shape_int(b)

    # Check that ranks are equal.
    if len(shape_a) != len(shape_b):
        raise ValueError('Inputs must have equal rank.')

    for i in range(len(shape_a)):
        a = tf.expand_dims(a, axis=2 * i + 1)
        b = tf.expand_dims(b, axis=2 * i)
    return tf.reshape(a * b, tuple(x * y for x, y in zip(shape_a, shape_b)))


@dispatch(TF)
def svd(a, compute_uv=True):
    res = tf.svd(a, full_matrices=True, compute_uv=compute_uv)
    return (res[1], res[0], res[2]) if compute_uv else res


@dispatch(TF)
def cholesky(a):
    return tf.cholesky(a)
