# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
from itertools import product
from numbers import Number

import nose.tools
import numpy as np
import tensorflow as tf
import torch
from plum import Dispatcher

from . import NP, TF, Torch

log = logging.getLogger('lab.' + __name__)
_dispatch = Dispatcher()

# Define some handy shorthands.
le = nose.tools.assert_less_equal
lt = nose.tools.assert_less
eq = nose.tools.assert_equal
neq = nose.tools.assert_not_equal
ge = nose.tools.assert_greater_equal
gt = nose.tools.assert_greater
raises = nose.tools.assert_raises
ok = nose.tools.ok_
approx = np.testing.assert_array_almost_equal


def call(f, method, args=(), res=True):
    eq(getattr(f, method)(*args), res)


def lam(f, args=()):
    ok(f(*args), 'Lambda returned False.')


@_dispatch({NP, Number})
def to_np(x):
    """Convert a tensor to NumPy."""
    return x


@_dispatch(tuple)
def to_np(tup):
    return tuple(to_np(x) for x in tup)


@_dispatch(Torch)
def to_np(x):
    return x.numpy()


@_dispatch(TF)
def to_np(x):
    with tf.Session() as sess:
        return sess.run(x)


@_dispatch(object, object)
def allclose(x, y):
    """Assert that two numeric objects are close."""
    np.testing.assert_allclose(to_np(x), to_np(y))


@_dispatch(tuple, tuple)
def allclose(x, y):
    eq(len(x), len(y))
    for xi, yi in zip(x, y):
        allclose(xi, yi)


def check_function(f, args_spec, kw_args_spec):
    """Check that a function produces consistent output."""
    # Construct product of keyword arguments.
    kw_args_prod = list(product(*[[(k, v) for v in vs.values()]
                                  for k, vs in kw_args_spec.items()]))
    kw_args_prod = [{k: v for k, v in kw_args} for kw_args in kw_args_prod]

    # Add default call.
    kw_args_prod += [{}]

    # Construct product of arguments.
    args_prod = list(product(*[arg.forms() for arg in args_spec]))

    # Check consistency of results.
    for kw_args in kw_args_prod:
        # Compare everything against the first result.
        first_result = f(*args_prod[0], **kw_args)

        for args in args_prod:
            # Skip mixes of TF and Torch.
            any_tf = any(isinstance(arg, TF) for arg in args)
            any_torch = any(isinstance(arg, Torch) for arg in args)
            if any_tf and any_torch:
                continue

            log.debug('Call with arguments "{}" and keyword arguments "{}".'
                      ''.format(args, kw_args))
            allclose(first_result, f(*args, **kw_args))


class Tensor(object):
    """Tensor placeholder for in argument specification."""

    def __init__(self, *dims, **kw_args):
        if 'mat' not in kw_args or kw_args['mat'] is None:
            self.mat = np.random.randn(*dims)
        else:
            self.mat = kw_args['mat']

    def forms(self):
        return [self.np(), self.tf(), self.torch()]

    def np(self):
        return self.mat

    def tf(self):
        return tf.constant(self.mat)

    def torch(self):
        return torch.tensor(self.mat)


class Matrix(Tensor):
    """Matrix placeholder for in argument specification."""

    def __init__(self, rows=3, cols=None, mat=None):
        # Default the number of columns to the number of rows.
        cols = rows if cols is None else cols
        Tensor.__init__(self, rows, cols, mat=mat)


class PSD(Matrix):
    """Positive-definite matrix placeholder for in argument specification."""

    def __init__(self, rows=3):
        a = np.random.randn(rows, rows)
        Matrix.__init__(self, mat=np.matmul(a, np.transpose(a)))


class Value(object):
    """Value placeholder for in keyword argument specification."""

    def __init__(self, values):
        if not isinstance(values, (tuple, list)):
            self._values = [values]
        else:
            self._values = values

    def values(self):
        return self._values


class Bool(Value):
    """Boolean placeholder for in keyword argument specification."""

    def __init__(self):
        Value.__init__(self, [False, True])
