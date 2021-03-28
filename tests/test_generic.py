import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special
import tensorflow as tf
import torch
from plum import NotFoundLookupError

import lab as B

# noinspection PyUnresolvedReferences
from .util import (
    autograd_box,
    check_function,
    Tensor,
    Value,
    PositiveTensor,
    BoolTensor,
    NaNTensor,
    Bool,
    allclose,
    check_lazy_shapes,
)


def test_constants(check_lazy_shapes):
    assert B.pi == np.pi
    assert B.log_2_pi == np.log(2 * np.pi)
    assert B.nan is np.nan


def test_isnan(check_lazy_shapes):
    check_function(B.isnan, (NaNTensor(),), {}, assert_dtype=False)
    check_function(B.isnan, (NaNTensor(2),), {}, assert_dtype=False)
    check_function(B.isnan, (NaNTensor(2, 3),), {}, assert_dtype=False)


def test_device_and_move_to_active_device(check_lazy_shapes):
    for a in Tensor(2, 2).forms():
        assert "cpu" in str(B.device(a)).lower()
        allclose(B.move_to_active_device(a), a)


@pytest.mark.parametrize("t", [tf.float32, torch.float32, jnp.float32])
@pytest.mark.parametrize(
    "f",
    [
        lambda t: B.zeros(t, 2, 2),
        lambda t: B.ones(t, 2, 2),
        lambda t: B.eye(t, 2),
        lambda t: B.linspace(t, 0, 5, 10),
        lambda t: B.range(t, 10),
        lambda t: B.rand(t, 10),
        lambda t: B.randn(t, 10),
    ],
)
def test_device_placement(f, t, check_lazy_shapes):
    # Check that explicit allocation on CPU works.
    with B.device("cpu"):
        f(t)

    # Check that allocation on a non-existing device breaks.
    with pytest.raises(Exception):
        with B.device("magic-device"):
            f(t)

    # Reset the active device. This is still set to "magic-device" due to the above
    # test.
    B.Device.active_name = None


def test_move_to_active_device_jax(check_lazy_shapes):
    a = jnp.ones(2)

    # No device specified: should do nothing.
    assert B.move_to_active_device(a) is a

    # Move to CPU without identifier.
    with B.device("cpu"):
        assert B.move_to_active_device(a) is not a
        allclose(B.move_to_active_device(a), a)

    # Move to CPU with identifier. Also check that capitalisation does not matter.
    with B.device("CPU:0"):
        assert B.move_to_active_device(a) is not a
        allclose(B.move_to_active_device(a), a)

    # Give invalid syntax.
    B.Device.active_name = "::"
    with pytest.raises(ValueError):
        B.move_to_active_device(a)
    B.Device.active_name = None


@pytest.mark.parametrize("f", [B.zeros, B.ones, B.eye])
def test_zeros_ones_eye(f, check_lazy_shapes):
    # Check consistency.
    check_function(
        f,
        (Value(np.float32, tf.float32, torch.float32, jnp.float32), Value(2), Value(3)),
    )

    # Check shape of calls.
    assert B.shape(f(2)) == (2, 2) if f is B.eye else (2,)
    assert B.shape(f(2, 3)) == (2, 3)
    assert B.shape(f(2, 3, 4)) == (2, 3, 4)

    # Check type of calls.
    assert B.dtype(f(2)) == B.default_dtype
    assert B.dtype(f(2, 3)) == B.default_dtype
    assert B.dtype(f(2, 3, 4)) == B.default_dtype

    # Specify a data type:
    for t1, t2 in [
        (np.float32, np.int64),
        (tf.float32, tf.int64),
        (torch.float32, torch.int64),
        (jnp.float32, jnp.int64),
    ]:
        # Check shape of calls.
        assert B.shape(f(t2, 2)) == (2, 2) if f is B.eye else (2,)
        assert B.shape(f(t2, 2, 3)) == (2, 3)
        assert B.shape(f(t2, 2, 3, 4)) == (2, 3, 4)

        # Check type of calls.
        assert B.dtype(f(t2, 2)) is t2
        assert B.dtype(f(t2, 2, 3)) is t2
        assert B.dtype(f(t2, 2, 3, 4)) is t2

        # Check reference calls.
        for ref in [B.randn(t1, 4, 5), B.randn(t1, 3, 4, 5)]:
            assert B.shape(f(ref)) == B.shape(ref)
            assert B.dtype(f(ref)) is t1


@pytest.mark.parametrize("f", [B.zero, B.one])
def test_zero_one(f, check_lazy_shapes):
    # Check consistency.
    check_function(f, (Value(np.float32, tf.float32, torch.float32, jnp.float32),))

    # Check reference calls.
    for t in [np.float32, tf.float32, torch.float32, jnp.float32]:
        assert B.dtype(f(B.randn(t))) is t


def test_linspace(check_lazy_shapes):
    # Check correctness.
    allclose(B.linspace(0, 1, 10), np.linspace(0, 1, 10, dtype=B.default_dtype))

    # Check consistency.
    check_function(
        B.linspace,
        (
            Value(np.float32, tf.float32, torch.float32, jnp.float32),
            Value(0),
            Value(10),
            Value(20),
        ),
    )


def test_range(check_lazy_shapes):
    # Check correctness.
    allclose(B.range(5), np.arange(5))
    allclose(B.range(2, 5), np.arange(2, 5))
    allclose(B.range(2, 5, 2), np.arange(2, 5, 2))

    # Check various step sizes.
    for step in [1, 1.0, 0.25]:
        check_function(
            B.range,
            (
                Value(np.float32, tf.float32, torch.float32, jnp.float32),
                Value(0),
                Value(5),
                Value(step),
            ),
        )

    # Check two-argument specification.
    check_function(
        B.range,
        (Value(np.float32, tf.float32, torch.float32, jnp.float32), Value(0), Value(5)),
    )

    # Check one-argument specification.
    check_function(
        B.range, (Value(np.float32, tf.float32, torch.float32, jnp.float32), Value(5))
    )


def test_cast(check_lazy_shapes):
    # Test casting to a given data type.
    assert B.dtype(B.cast(np.float64, 1)) is np.float64
    assert B.dtype(B.cast(np.float64, np.array(1))) is np.float64
    assert B.dtype(B.cast(np.float64, autograd_box(np.float32(1)))) is np.float64

    assert B.dtype(B.cast(tf.float64, 1)) is tf.float64
    assert B.dtype(B.cast(tf.float64, np.array(1))) is tf.float64
    assert B.dtype(B.cast(tf.float64, tf.constant(1))) is tf.float64

    assert B.dtype(B.cast(torch.float64, 1)) is torch.float64
    assert B.dtype(B.cast(torch.float64, np.array(1))) is torch.float64
    assert B.dtype(B.cast(torch.float64, torch.tensor(1))) is torch.float64

    assert B.dtype(B.cast(jnp.float64, 1)) is jnp.float64
    assert B.dtype(B.cast(jnp.float64, np.array(1))) is jnp.float64
    assert B.dtype(B.cast(jnp.float64, jnp.array(1))) is jnp.float64


@pytest.mark.parametrize(
    "x",
    [
        1,
        np.float32(1),
        B.randn(np.float32),
        autograd_box(B.randn(np.float32)),
        B.randn(tf.float32),
        B.randn(torch.float32),
        B.randn(jnp.float32),
    ],
)
def test_cast_own_dtype(x, check_lazy_shapes):
    # Test that casting to its own data type does nothing.
    assert x is B.cast(B.dtype(x), x)


@pytest.mark.parametrize("dtype", [np.float64, tf.float64, torch.float64, jnp.float64])
def test_cast_shape_element(dtype, check_lazy_shapes):
    assert B.dtype(B.cast(dtype, B.shape(B.ones(dtype, 1))[0])) is dtype


@pytest.mark.parametrize(
    "f",
    [
        B.identity,
        B.negative,
        B.abs,
        B.sign,
        B.exp,
        B.sin,
        B.cos,
        B.tan,
        B.tanh,
        B.erf,
        B.sigmoid,
        B.softplus,
        B.relu,
    ],
)
def test_unary_signed(f, check_lazy_shapes):
    check_function(f, (Tensor(),))
    check_function(f, (Tensor(2),))
    check_function(f, (Tensor(2, 3),))


@pytest.mark.parametrize("f", [B.log, B.sqrt])
def test_unary_positive(f, check_lazy_shapes):
    check_function(f, (PositiveTensor(),))
    check_function(f, (PositiveTensor(2),))
    check_function(f, (PositiveTensor(2, 3),))


@pytest.mark.parametrize("a", [0, -1, 1])
def test_softplus_correctness(a, check_lazy_shapes):
    allclose(B.softplus(a), np.log(1 + np.exp(a)))


@pytest.mark.parametrize(
    "f", [B.add, B.subtract, B.multiply, B.divide, B.minimum, B.maximum, B.leaky_relu]
)
def test_binary_signed(f, check_lazy_shapes):
    check_function(f, (Tensor(), Tensor()))
    check_function(f, (Tensor(2), Tensor(2)))
    check_function(f, (Tensor(2, 3), Tensor(2, 3)))


@pytest.mark.parametrize("f", [B.power])
def test_binary_positive_first(f, check_lazy_shapes):
    check_function(f, (PositiveTensor(), Tensor()))
    check_function(f, (PositiveTensor(2), Tensor(2)))
    check_function(f, (PositiveTensor(2, 3), Tensor(2, 3)))


@pytest.mark.parametrize("f", [B.min, B.max, B.sum, B.mean, B.std, B.logsumexp])
def test_reductions(f, check_lazy_shapes):
    check_function(f, (Tensor(),))
    check_function(f, (Tensor(2),))
    check_function(f, (Tensor(2),), {"axis": Value(0)})
    check_function(f, (Tensor(2, 3),))
    check_function(f, (Tensor(2, 3),), {"axis": Value(0, 1)})


def test_logsumexp_correctness(check_lazy_shapes):
    mat = PositiveTensor(3, 4).np()
    allclose(B.logsumexp(mat, axis=1), scipy.special.logsumexp(mat, axis=1))


@pytest.mark.parametrize("f", [B.all, B.any])
def test_logical_reductions(f, check_lazy_shapes):
    check_function(f, (BoolTensor(),), {}, assert_dtype=False)
    check_function(f, (BoolTensor(2),), {}, assert_dtype=False)
    check_function(f, (BoolTensor(2),), {"axis": Value(0)}, assert_dtype=False)
    check_function(f, (BoolTensor(2, 3),), {}, assert_dtype=False)
    check_function(f, (BoolTensor(2, 3),), {"axis": Value(0, 1)}, assert_dtype=False)


@pytest.mark.parametrize("f", [B.lt, B.le, B.gt, B.ge])
def test_logical_comparisons(f, check_lazy_shapes):
    check_function(f, (Tensor(), Tensor()), {}, assert_dtype=False)
    check_function(f, (Tensor(2), Tensor(2)), {}, assert_dtype=False)
    check_function(f, (Tensor(2, 3), Tensor(2, 3)), {}, assert_dtype=False)


def test_bvn_cdf(check_lazy_shapes):
    check_function(
        B.bvn_cdf,
        (PositiveTensor(5), PositiveTensor(5), PositiveTensor(5)),
        {},
        assert_dtype=False,
    )


def test_cond(check_lazy_shapes):
    def f(v, x):
        return B.cond(v > 0, lambda y: 2 * y, lambda y: y ** 2, x)

    for _ in range(10):
        check_function(f, (Tensor(), Tensor(4)))


def test_scan(check_lazy_shapes):
    # Check consistency by inputting various shapes for a simple scanning
    # function.

    def scan_f(prev, x):
        return prev

    check_function(B.scan, (Value(scan_f), Tensor(4), Tensor()))
    check_function(B.scan, (Value(scan_f), Tensor(4), Tensor(2)))
    check_function(B.scan, (Value(scan_f), Tensor(4), Tensor(2, 3)))
    check_function(B.scan, (Value(scan_f), Tensor(4, 5), Tensor()))
    check_function(B.scan, (Value(scan_f), Tensor(4, 5), Tensor(2)))
    check_function(B.scan, (Value(scan_f), Tensor(4, 5), Tensor(2, 3)))

    # Check correctness by comparing NumPy against TensorFlow for more
    # complicated scanning function.

    def scan_f(prev, x):
        prev_h, _ = prev
        h = prev_h * x + 1
        y = 2 * h + x
        return h, y

    xs = Tensor(10, 3, 4)
    init_h = Tensor(3, 4)
    init_y = Tensor(3, 4)
    allclose(
        B.scan(scan_f, xs.np(), init_h.np(), init_y.np()),
        B.scan(scan_f, xs.tf(), init_h.tf(), init_y.tf()),
    )

    # Check shape checking.

    def incorrect_scan_f(prev, x):
        return prev + prev

    with pytest.raises(RuntimeError):
        B.scan(incorrect_scan_f, Tensor(4).torch(), Tensor().torch(), Tensor().torch())


def test_sort(check_lazy_shapes):
    check_function(B.sort, (Tensor(4),), {"axis": Value(-1, 0), "descending": Bool()})
    # AutoGrad cannot sort multidimensional arrays.
    check_function(
        B.sort,
        (Tensor(2, 3, 4),),
        {"axis": Value(-1, 0, 1, 2), "descending": Bool()},
        skip=[B.AGNumeric],
    )


def test_argsort(check_lazy_shapes):
    check_function(
        B.argsort,
        (Tensor(4),),
        {"axis": Value(-1, 0), "descending": Bool()},
        assert_dtype=False,
    )
    # AutoGrad cannot sort multidimensional arrays.
    check_function(
        B.argsort,
        (Tensor(2, 3, 4),),
        {"axis": Value(-1, 0, 1, 2), "descending": Bool()},
        skip=[B.AGNumeric],
        assert_dtype=False,
    )


def test_to_numpy(check_lazy_shapes):
    check_function(B.to_numpy, (Tensor(),))
    check_function(B.to_numpy, (Tensor(4),))


def test_to_numpy_multiple_objects(check_lazy_shapes):
    assert B.to_numpy(tf.constant(1), tf.constant(1)) == (1, 1)


def test_to_numpy_list(check_lazy_shapes):
    x = B.to_numpy([tf.constant(1)])
    assert isinstance(x[0], (B.Number, B.NPNumeric))


def test_to_numpy_tuple(check_lazy_shapes):
    x = B.to_numpy((tf.constant(1),))
    assert isinstance(x[0], (B.Number, B.NPNumeric))


def test_to_numpy_dict(check_lazy_shapes):
    x = B.to_numpy({"a": tf.constant(1)})
    assert isinstance(x["a"], (B.Number, B.NPNumeric))
