import warnings

import numpy  as np
import pytest
import tensorflow as tf
import torch

import lab as B
from .util import (
    Tensor,
    allclose,
    dtype_equal,
    to_np
)


@pytest.mark.parametrize('dtype', [np.float32, tf.float32, torch.float32])
def test_set_seed(dtype):
    B.set_random_seed(0)
    x = to_np(B.rand(dtype))
    B.set_random_seed(0)
    y = to_np(B.rand(dtype))
    assert x == y


@pytest.mark.parametrize('f', [B.rand, B.randn])
def test_random_generators(f):
    # Test without specifying data type.
    dtype_equal(B.dtype(f()), B.default_dtype)
    assert B.shape(f()) == ()
    dtype_equal(B.dtype(f(2)), B.default_dtype)
    allclose(B.shape(f(2)), (2,))
    dtype_equal(B.dtype(f(2, 3)), B.default_dtype)
    assert B.shape(f(2, 3)) == (2, 3)

    # Test with specifying data type.
    for t in [np.float32, tf.float32, torch.float32]:
        # Test direct specification.
        dtype_equal(B.dtype(f(t)), t)
        assert B.shape(f(t)) == ()
        dtype_equal(B.dtype(f(t, 2)), t)
        assert B.shape(f(t, 2)) == (2,)
        dtype_equal(B.dtype(f(t, 2, 3)), t)
        assert B.shape(f(t, 2, 3)) == (2, 3)

        # Test reference specification.
        dtype_equal(B.dtype(f(f(t))), t)
        assert B.shape(f(f())) == ()
        dtype_equal(B.dtype(f(f(t, 2))), t)
        assert B.shape(f(f(t, 2))) == (2,)
        dtype_equal(B.dtype(f(f(t, 2, 3))), t)
        assert B.shape(f(f(t, 2, 3))) == (2, 3)


@pytest.mark.parametrize('f', [B.rand, B.randn])
def test_conversion_warnings(f):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')

        # Trigger the warning!
        f(int, 5)

        assert len(w) == 1


def test_choice():
    # TODO: Can we use a parametrised test here?
    for x in Tensor(2).forms() + Tensor(2, 3).forms() + Tensor(2, 3, 4).forms():
        # Check shape.
        assert B.shape(B.choice(x)) == B.shape(x)[1:]
        assert B.shape(B.choice(x, 1)) == B.shape(x)[1:]
        assert B.shape(B.choice(x, 5))[0] == 5
        assert B.shape(B.choice(x, 5))[1:] == B.shape(x)[1:]

        # Check correctness.
        dtype = B.dtype(x)
        choices = set(to_np(B.choice(B.range(dtype, 5), 1000)))
        assert choices == set(to_np(B.range(dtype, 5)))
