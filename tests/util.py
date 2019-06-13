# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
from itertools import product

import numpy as np
import plum
import tensorflow as tf
import torch
from plum import Dispatcher

import lab as B

__all__ = ['dtype_equal',
           'to_np',
           'allclose',
           'check_function',
           'Tensor', 'PositiveTensor', 'BoolTensor', 'NaNTensor',
           'Matrix', 'PSD', 'PSDTriangular',
           'Tuple', 'List', 'Value', 'Bool']

log = logging.getLogger('lab.' + __name__)

_dispatch = Dispatcher()

approx = np.testing.assert_array_almost_equal


def dtype_equal(x, y):
    # NumPy has two representations of data types, and TensorFlow data types
    # equal to NumPy data types.
    if isinstance(x, (type, np.dtype)) and isinstance(y, (type, np.dtype)):
        # We are safe to equate, because `x` nor `y` can be a TensorFlow or
        # PyTorch data type.
        assert x == y
    else:
        assert x is y


@_dispatch({B.NPNumeric, B.Number, B.Bool})
def to_np(x):
    """Convert a tensor to NumPy."""
    return x


@_dispatch({B.TorchNumeric, B.TFNumeric})
def to_np(x):
    return x.numpy()


@_dispatch({tuple, tf.TensorShape, torch.Size})
def to_np(tup):
    return tuple(to_np(x) for x in tup)


@_dispatch(list)
def to_np(lst):
    return to_np(tuple(lst))


@_dispatch(object, object, [bool])
def allclose(x, y, assert_dtype=False):
    """Assert that two numeric objects are close."""
    x, y = to_np(x), to_np(y)

    # Assert that data types are equal if it concerns floats.
    if assert_dtype:
        assert np.array(x).dtype == np.array(y).dtype

    np.testing.assert_allclose(x, y)


@_dispatch(tuple, tuple, [bool])
def allclose(x, y, assert_dtype=False):
    assert len(x) == len(y)
    for xi, yi in zip(x, y):
        allclose(xi, yi, assert_dtype)


def check_function(f, args_spec, kw_args_spec=None, assert_dtype=True):
    """Check that a function produces consistent output."""
    if kw_args_spec is None:
        kw_args_spec = {}

    # Construct product of keyword arguments.
    kw_args_prod = list(product(*[[(k, v) for v in vs.forms()]
                                  for k, vs in kw_args_spec.items()]))
    kw_args_prod = [{k: v for k, v in kw_args} for kw_args in kw_args_prod]

    # Add default call.
    kw_args_prod += [{}]

    # Construct product of arguments.
    args_prod = list(product(*[arg.forms() for arg in args_spec]))

    # Construct types TF and Torch numerics, lists, or tuples.
    tf_type = plum.Union(B.TFNumeric,
                         plum.List(B.TFNumeric),
                         plum.Tuple(B.TFNumeric))
    torch_type = plum.Union(B.TorchNumeric,
                            plum.List(B.TorchNumeric),
                            plum.Tuple(B.TorchNumeric))

    # Check consistency of results.
    for kw_args in kw_args_prod:
        # Compare everything against the first result.
        first_result = f(*args_prod[0], **kw_args)

        for args in args_prod:
            # Skip mixes of TF and Torch numerics, lists, or tuples.
            any_tf = any(isinstance(arg, tf_type) for arg in args)
            any_torch = any(isinstance(arg, torch_type) for arg in args)
            if any_tf and any_torch:
                log.debug('Skipping call with arguments {} and keyword '
                          'arguments {}.'.format(args, kw_args))
                continue

            log.debug('Call with arguments {} and keyword arguments {}.'
                      ''.format(args, kw_args))
            allclose(first_result, f(*args, **kw_args), assert_dtype)


class Tensor(object):
    """Tensor placeholder."""

    def __init__(self, *dims, **kw_args):
        if 'mat' not in kw_args or kw_args['mat'] is None:
            self.mat = np.array(np.random.randn(*dims))
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


class PositiveTensor(Tensor):
    """Positive tensor placeholder."""

    def __init__(self, *dims, **kw_args):
        if 'mat' not in kw_args or kw_args['mat'] is None:
            mat = np.array(np.random.rand(*dims))
        else:
            mat = kw_args['mat']
        Tensor.__init__(self, mat=mat)


class BoolTensor(Tensor):
    """Boolean tensor placeholder."""

    def __init__(self, *dims, **kw_args):
        if 'mat' not in kw_args or kw_args['mat'] is None:
            mat = np.array(np.random.rand(*dims) > .5)
        else:
            mat = kw_args['mat']
        Tensor.__init__(self, mat=mat)

    def torch(self):
        return torch.tensor(self.mat.astype(np.uint8))


class NaNTensor(Tensor):
    """Tensor containing NaNs placeholder."""

    def __init__(self, *dims, **kw_args):
        if 'mat' not in kw_args or kw_args['mat'] is None:
            mat = np.array(np.random.randn(*dims))
            set_nan = np.array(np.random.rand(*dims) > .5)
            mat[set_nan] = np.nan
        else:
            mat = kw_args['mat']
        Tensor.__init__(self, mat=mat)


class Matrix(Tensor):
    """Matrix placeholder."""

    def __init__(self, *shape, **kw_args):
        # Handle shorthands.
        if shape == ():
            shape = (3, 3)
        elif len(shape) == 1:
            shape = shape * 2

        Tensor.__init__(self, *shape, **kw_args)


class PSD(Matrix):
    """Positive-definite tensor placeholder."""

    def __init__(self, *shape):
        # Handle shorthands.
        if shape == ():
            shape = (3, 3)
        elif len(shape) == 1:
            shape = shape * 2

        if not shape[-2] == shape[-1]:
            raise ValueError('PSD matrix must be square.')

        a = np.random.randn(*shape)
        perm = list(range(len(a.shape)))
        perm[-2], perm[-1] = perm[-1], perm[-2]
        a_t = np.transpose(a, perm)
        Matrix.__init__(self, mat=np.matmul(a, a_t))


class PSDTriangular(PSD):
    def __init__(self, *shape, **kw_args):
        PSD.__init__(self, *shape)

        # Zero upper triangular part.
        for i in range(self.mat.shape[0]):
            for j in range(i + 1, self.mat.shape[1]):
                self.mat[..., i, j] = 0

        # Create upper-triangular matrices, if asked for.
        if kw_args.get('upper', False):
            perm = list(range(len(self.mat.shape)))
            perm[-2], perm[-1] = perm[-1], perm[-2]
            self.mat = np.transpose(self.mat, perm)


class Tuple(object):
    """Tuple placeholder."""

    def __init__(self, *xs):
        self.xs = xs

    def forms(self):
        return tuple(zip(*(x.forms() for x in self.xs)))


class List(object):
    """List placeholder for in argument specification."""

    def __init__(self, *xs):
        self.xs = xs

    def forms(self):
        return list(zip(*(x.forms() for x in self.xs)))


class Value(object):
    """Value placeholder."""

    def __init__(self, *values):
        self._values = values

    def forms(self):
        return self._values


class Bool(Value):
    """Boolean placeholder."""

    def __init__(self):
        Value.__init__(self, False, True)
