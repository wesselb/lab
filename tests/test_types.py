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
from .util import check_lazy_shapes, autograd_box


def test_numeric(check_lazy_shapes):
    # Test convenient types.
    assert isinstance(1, B.Int)
    assert isinstance(np.int32(1), B.Int)
    assert isinstance(np.uint64(1), B.Int)

    assert isinstance(1.0, B.Float)
    assert isinstance(np.float32(1), B.Float)

    assert isinstance(1 + 0j, B.Complex)
    assert isinstance(np.complex64(1), B.Complex)

    assert isinstance(True, B.Bool)
    assert isinstance(np.bool_(True), B.Bool)

    assert isinstance(np.uint(1), B.Number)
    assert isinstance(np.float64(1), B.Number)
    assert isinstance(np.complex64(1), B.Number)

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

    # `torch.bool` has a manual addition, so test it separately.
    assert convert(torch.bool, B.NPDType) is bool


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

    # Test tuples, which promote.
    assert B.dtype(1, 1) is np.int64
    assert B.dtype((1, 1)) is np.int64
    assert B.dtype(1, 1.0) is np.float64
    assert B.dtype((1, 1.0)) is np.float64


def test_issubdtype(check_lazy_shapes):
    assert B.issubdtype(np.float32, np.floating)
    assert B.issubdtype(tf.float32, np.floating)
    assert B.issubdtype(torch.float32, np.floating)
    assert B.issubdtype(jnp.float32, np.floating)
    assert not B.issubdtype(np.float32, np.integer)
    assert not B.issubdtype(tf.float32, np.integer)
    assert not B.issubdtype(torch.float32, np.integer)
    assert not B.issubdtype(jnp.float32, np.integer)


def test_promote_dtypes(check_lazy_shapes):
    # Check one-argument case.
    assert B.promote_dtypes(int) is int
    assert B.promote_dtypes(float) is float

    # Check multi-argument case.
    for t_int, t_float in [
        (np.int64, np.float64),
        (tf.int64, tf.float64),
        (torch.int64, torch.float64),
        (jnp.int64, jnp.float64),
    ]:
        # Also check that the conversion back is right.
        assert B.promote_dtypes(t_int, int) is t_int
        assert B.promote_dtypes(t_int, int, int) is t_int
        assert B.promote_dtypes(t_int, float) is t_float
        assert B.promote_dtypes(t_int, int, float) is t_float


def test_dtype_float(check_lazy_shapes):
    assert B.dtype_float(np.float32) is np.float32
    assert B.dtype_float(np.float32(1)) is np.float32
    assert B.dtype_float(np.float64) is np.float64
    assert B.dtype_float(np.float64(1)) is np.float64
    assert B.dtype_float(int) is np.float64
    assert B.dtype_float(1) is np.float64


def test_dtype_int(check_lazy_shapes):
    assert B.dtype_int(np.float32) is np.int32
    assert B.dtype_int(np.float32(1)) is np.int32
    assert B.dtype_int(np.float64) is np.int64
    assert B.dtype_int(np.float64(1)) is np.int64
    assert B.dtype_int(int) is int
    assert B.dtype_int(1) is int
    # Test conversion back to right framework type. This conversion is thoroughly
    # tested for `B.promote_dtypes`.
    assert B.dtype_int(tf.float32) is tf.int32
    assert B.dtype_int(tf.constant(1.0, dtype=tf.float32)) is tf.int32
    assert B.dtype_int(tf.float64) is tf.int64
    assert B.dtype_int(tf.constant(1.0, dtype=tf.float64)) is tf.int64


@pytest.mark.parametrize(
    "t, FWRandomState",
    [
        (np.float64, B.NPRandomState),
        (tf.float64, B.TFRandomState),
        (torch.float64, B.TorchRandomState),
        (jnp.float64, B.JAXRandomState),
    ],
)
def test_random_state(t, FWRandomState, check_lazy_shapes):
    assert isinstance(B.create_random_state(t), FWRandomState)


def test_random_state_jax(check_lazy_shapes):
    # Splitting a JAX random state gives a NumPy array.
    assert isinstance(np.array(1), B.JAXRandomState)


@pytest.mark.parametrize(
    "t, FWDevice",
    [
        (tf.float64, B.TFDevice),
        (torch.float64, B.TorchDevice),
        (jnp.float64, B.JAXDevice),
    ],
)
def test_device(t, FWDevice, check_lazy_shapes):
    a = B.randn(t, 2, 2)
    assert isinstance(B.device(a), FWDevice)
    assert isinstance(B.device(a), B.Device)

    # Test conversion to string.
    assert isinstance(convert(B.device(a), str), str)


@pytest.mark.parametrize("t", [B.NP, B.Framework])
def test_framework_np(t, check_lazy_shapes):
    assert isinstance(np.array(1), t)
    assert isinstance(np.float32, t)
    assert isinstance(B.create_random_state(np.float32), t)


@pytest.mark.parametrize("t", [B.AG, B.Framework])
def test_framework_ag(t, check_lazy_shapes):
    assert isinstance(autograd_box(np.array(1)), t)
    assert isinstance(np.float32, t)
    assert isinstance(B.create_random_state(np.float32), t)


@pytest.mark.parametrize("t", [B.TF, B.Framework])
def test_framework_tf(t, check_lazy_shapes):
    assert isinstance(tf.constant(1), t)
    assert isinstance(tf.float32, t)
    assert isinstance(B.create_random_state(tf.float32), t)
    assert isinstance(B.device(tf.constant(1)), t)


@pytest.mark.parametrize("t", [B.Torch, B.Framework])
def test_framework_torch(t, check_lazy_shapes):
    assert isinstance(torch.tensor(1), t)
    assert isinstance(torch.float32, t)
    assert isinstance(B.create_random_state(torch.float32), t)
    assert isinstance(B.device(torch.tensor(1)), t)


@pytest.mark.parametrize("t", [B.JAX, B.Framework])
def test_framework_jax(t, check_lazy_shapes):
    assert isinstance(jnp.asarray(1), t)
    assert isinstance(jnp.float32, t)
    assert isinstance(B.create_random_state(jnp.float32), t)
    assert isinstance(B.device(jnp.asarray(1)), t)
