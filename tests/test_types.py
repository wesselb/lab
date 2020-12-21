import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch
from autograd import grad
from plum.promotion import _promotion_rule, convert

import lab as B

# noinspection PyUnresolvedReferences
from .util import check_lazy_shapes


def test_numeric(check_lazy_shapes):
    # Test convenient types.
    assert isinstance(1, B.Int)
    assert isinstance(np.int32(1), B.Int)
    assert isinstance(np.uint64(1), B.Int)
    assert isinstance(1.0, B.Float)
    assert isinstance(np.float32(1), B.Float)
    assert isinstance(True, B.Bool)
    assert isinstance(np.bool_(True), B.Bool)
    assert isinstance(np.uint(1), B.Number)
    assert isinstance(np.float64(1), B.Number)

    # Test NumPy.
    assert isinstance(np.array(1), B.NPNumeric)

    # Test TensorFlow.
    assert isinstance(tf.constant(1), B.TFNumeric)
    assert isinstance(tf.Variable(1), B.TFNumeric)

    # Test Torch.
    assert isinstance(torch.tensor(1), B.TorchNumeric)

    # Test JAX.
    assert isinstance(jnp.array(1), B.JAXNumeric)

    # Test general numeric type.
    assert isinstance(1, B.Numeric)
    assert isinstance(np.bool_(1), B.Numeric)
    assert isinstance(np.float64(1), B.Numeric)
    assert isinstance(np.array(1), B.Numeric)
    assert isinstance(tf.constant(1), B.Numeric)
    assert isinstance(torch.tensor(1), B.Numeric)

    # Test promotion.
    assert _promotion_rule(np.array(1), tf.constant(1)) == B.TFNumeric
    assert _promotion_rule(np.array(1), tf.Variable(1)) == B.TFNumeric
    assert _promotion_rule(tf.constant(1), tf.Variable(1)) == B.TFNumeric
    assert _promotion_rule(np.array(1), torch.tensor(1)) == B.TorchNumeric
    assert _promotion_rule(np.array(1), jnp.array(1)) == B.JAXNumeric
    with pytest.raises(TypeError):
        _promotion_rule(B.TFNumeric, B.TorchNumeric)

    # Test conversion.
    assert isinstance(convert(np.array(1), B.TFNumeric), B.TFNumeric)
    assert isinstance(convert(np.array(1), B.TorchNumeric), B.TorchNumeric)
    assert isinstance(convert(np.array(1), B.JAXNumeric), B.JAXNumeric)


def test_autograd_tracing(check_lazy_shapes):
    found_objs = []

    def f(x):
        found_objs.append(x)
        return B.sum(x)

    # Test that function runs.
    f(np.ones(5))
    found_objs[:] = []  # Clear found objects.

    # Catch AutoGrad object.
    grad(f)(np.ones(5))

    # Test that objects are of the right type.
    for obj in found_objs:
        assert isinstance(obj, B.AGNumeric)


def test_jax_tracing(check_lazy_shapes):
    found_objs = []

    def f(x):
        found_objs.append(x)
        return B.sum(x)

    # Catch JAX object during JIT and during gradient computation.
    jax.grad(f)(np.ones(5))
    jax.jit(f)(np.ones(5))

    # Test that objects are of the right type.
    for obj in found_objs:
        assert isinstance(obj, B.JAXNumeric)


def test_data_type(check_lazy_shapes):
    assert isinstance(np.float32, B.NPDType)
    assert isinstance(np.float32, B.DType)
    assert isinstance(tf.float32, B.TFDType)
    assert isinstance(tf.float32, B.DType)
    assert isinstance(torch.float32, B.TorchDType)
    assert isinstance(torch.float32, B.DType)
    assert isinstance(jnp.float32, B.JAXDType)
    assert isinstance(jnp.float32, B.DType)

    # Check that the AutoGrad and JAX data types are just the NumPy data type. Then
    # there is nothing left to check.
    assert B.AGDType == B.NPDType

    # Test conversion between data types.
    assert convert(np.float32, B.TFDType) is tf.float32
    assert convert(np.float32, B.TorchDType) is torch.float32
    assert convert(np.float32, B.JAXDType) is jnp.float32
    assert convert(tf.float32, B.NPDType) is np.float32
    assert convert(tf.float32, B.TorchDType) is torch.float32
    assert convert(tf.float32, B.JAXDType) is jnp.float32
    assert convert(torch.float32, B.NPDType) is np.float32
    assert convert(torch.float32, B.TFDType) is tf.float32
    assert convert(torch.float32, B.JAXDType) is jnp.float32
    assert convert(jnp.float32, B.NPDType) is np.float32
    assert convert(jnp.float32, B.TFDType) is tf.float32
    assert convert(jnp.float32, B.TorchDType) is torch.float32


def test_issubdtype(check_lazy_shapes):
    assert B.issubdtype(np.float32, np.floating)
    assert B.issubdtype(tf.float32, np.floating)
    assert B.issubdtype(torch.float32, np.floating)
    assert B.issubdtype(jnp.float32, np.floating)
    assert not B.issubdtype(np.float32, np.integer)
    assert not B.issubdtype(tf.float32, np.integer)
    assert not B.issubdtype(torch.float32, np.integer)
    assert not B.issubdtype(jnp.float32, np.integer)


def test_dtype(check_lazy_shapes):
    assert B.dtype(1) is int
    assert B.dtype(1.0) is float
    assert B.dtype(np.array(1, dtype=np.int32)) is np.int32
    assert B.dtype(np.array(1.0, dtype=np.float32)) is np.float32
    assert B.dtype(tf.constant(1, dtype=tf.int32)) is tf.int32
    assert B.dtype(tf.constant(1.0, dtype=tf.float32)) is tf.float32
    assert B.dtype(torch.tensor(1, dtype=torch.int32)) is torch.int32
    assert B.dtype(torch.tensor(1.0, dtype=torch.float32)) is torch.float32
    assert B.dtype(jnp.array(1, dtype=jnp.int32)) is jnp.int32
    assert B.dtype(jnp.array(1.0, dtype=jnp.float32)) is jnp.float32


@pytest.mark.parametrize("t", [B.NP, B.Framework])
def test_framework_np(t, check_lazy_shapes):
    assert isinstance(np.array(1), t)
    assert isinstance(np.float32, t)


@pytest.mark.parametrize("t", [B.TF, B.Framework])
def test_framework_tf(t, check_lazy_shapes):
    assert isinstance(tf.constant(1), t)
    assert isinstance(tf.float32, t)


@pytest.mark.parametrize("t", [B.Torch, B.Framework])
def test_framework_torch(t, check_lazy_shapes):
    assert isinstance(torch.tensor(1), t)
    assert isinstance(torch.float32, t)


@pytest.mark.parametrize("t", [B.JAX, B.Framework])
def test_framework_jax(t, check_lazy_shapes):
    assert isinstance(jnp.asarray(1), t)
