import numpy as np
import pytest
import scipy.special
import tensorflow as tf
import torch
from plum import NotFoundLookupError

import lab as B
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
    dtype_equal
)


def test_constants():
    assert B.pi == np.pi
    assert B.log_2_pi == np.log(2 * np.pi)
    assert B.nan is np.nan


def test_isnan():
    check_function(B.isnan, (NaNTensor(),), {}, assert_dtype=False)
    check_function(B.isnan, (NaNTensor(2),), {}, assert_dtype=False)
    check_function(B.isnan, (NaNTensor(2, 3),), {}, assert_dtype=False)


@pytest.mark.parametrize('f', [B.zeros, B.ones, B.eye])
def test_zeros_ones_eye(f):
    # Check consistency.
    check_function(f, (Value(np.float32, tf.float32, torch.float32),
                       Value(2),
                       Value(3)))

    # Check shape of calls.
    assert B.shape(f(2)) == (2, 2) if f is B.eye else (2,)
    assert B.shape(f(2, 3)) == (2, 3)

    # Check type of calls.
    assert B.dtype(f(2)) == B.default_dtype
    assert B.dtype(f(2, 3)) == B.default_dtype

    # Check reference calls.
    for t1, t2 in [(np.float32, np.int64),
                   (tf.float32, tf.int64),
                   (torch.float32, torch.int64)]:
        ref = B.randn(t1, 4, 5)

        # Check shape of calls.
        assert B.shape(f(t2, 2)) == (2, 2) if f is B.eye else (2,)
        assert B.shape(f(t2, 2, 3)) == (2, 3)
        assert B.shape(f(ref)) == (4, 5)

        # Check type of calls.
        assert B.dtype(f(t2, 2)) == t2
        assert B.dtype(f(t2, 2, 3)) == t2
        assert B.dtype(f(ref)) == t1


@pytest.mark.parametrize('f', [B.zero, B.one])
def test_zero_one(f):
    # Check consistency.
    check_function(f, (Value(np.float32, tf.float32, torch.float32),))

    # Check reference calls.
    for t in [np.float32, tf.float32, torch.float32]:
        assert B.dtype(f(B.randn(t))) == t


def test_eye_exceptions():
    with pytest.raises(NotFoundLookupError):
        B.eye(3, 4, 5)
    for t in [np.float32, tf.float32, torch.float32]:
        with pytest.raises(NotFoundLookupError):
            B.eye(t, 3, 4, 5)


def test_linspace():
    # Check correctness.
    allclose(B.linspace(0, 1, 10), np.linspace(0, 1, 10, dtype=B.default_dtype))

    # Check consistency
    check_function(B.linspace, (Value(np.float32, tf.float32, torch.float32),
                                Value(0),
                                Value(10),
                                Value(20)))


def test_range():
    # Check correctness.
    allclose(B.range(5), np.arange(5))
    allclose(B.range(2, 5), np.arange(2, 5))
    allclose(B.range(2, 5, 2), np.arange(2, 5, 2))

    # Check various step sizes.
    for step in [1, 1.0, 0.25]:
        check_function(B.range, (Value(np.float32, tf.float32, torch.float32),
                                 Value(0),
                                 Value(5),
                                 Value(step)))

    # Check two-argument specification.
    check_function(B.range, (Value(np.float32, tf.float32, torch.float32),
                             Value(0),
                             Value(5)))

    # Check one-argument specification.
    check_function(B.range, (Value(np.float32, tf.float32, torch.float32),
                             Value(5)))


def test_cast():
    # Test casting to a given data type.
    dtype_equal(B.dtype(B.cast(np.float64, 1)), np.float64)
    dtype_equal(B.dtype(B.cast(np.float64, np.array(1))), np.float64)

    dtype_equal(B.dtype(B.cast(np.float64, autograd_box(np.float32(1)))),
                np.float64)

    dtype_equal(B.dtype(B.cast(tf.float64, 1)), tf.float64)
    dtype_equal(B.dtype(B.cast(tf.float64, np.array(1))), tf.float64)
    dtype_equal(B.dtype(B.cast(tf.float64, tf.constant(1))), tf.float64)

    dtype_equal(B.dtype(B.cast(torch.float64, 1)), torch.float64)
    dtype_equal(B.dtype(B.cast(torch.float64, np.array(1))), torch.float64)
    dtype_equal(B.dtype(B.cast(torch.float64, torch.tensor(1))), torch.float64)

    # Test that casting to its own data type does nothing.
    for x in [B.randn(np.float32),
              autograd_box(B.randn(np.float32)),
              B.randn(tf.float32),
              B.randn(torch.float32)]:
        assert x is B.cast(B.dtype(x), x)


@pytest.mark.parametrize('f', [B.identity,
                               B.negative,
                               B.abs,
                               B.sign,
                               B.exp,
                               B.sin,
                               B.cos,
                               B.tan,
                               B.tanh,
                               B.sigmoid,
                               B.softplus,
                               B.relu])
def test_unary_signed(f):
    check_function(f, (Tensor(),))
    check_function(f, (Tensor(2),))
    check_function(f, (Tensor(2, 3),))


@pytest.mark.parametrize('f', [B.log, B.sqrt])
def test_unary_positive(f):
    check_function(f, (PositiveTensor(),))
    check_function(f, (PositiveTensor(2),))
    check_function(f, (PositiveTensor(2, 3),))


@pytest.mark.parametrize('a', [0, -1, 1])
def test_softplus_correctness(a):
    allclose(B.softplus(a), np.log(1 + np.exp(a)))


@pytest.mark.parametrize('f', [B.add,
                               B.subtract,
                               B.multiply,
                               B.divide,
                               B.minimum,
                               B.maximum,
                               B.leaky_relu])
def test_binary_signed(f):
    check_function(f, (Tensor(), Tensor()))
    check_function(f, (Tensor(2), Tensor(2)))
    check_function(f, (Tensor(2, 3), Tensor(2, 3)))


@pytest.mark.parametrize('f', [B.power])
def test_binary_positive_first(f):
    check_function(f, (PositiveTensor(), Tensor()))
    check_function(f, (PositiveTensor(2), Tensor(2)))
    check_function(f, (PositiveTensor(2, 3), Tensor(2, 3)))


@pytest.mark.parametrize('f', [B.min, B.max, B.sum, B.mean, B.std, B.logsumexp])
def test_reductions(f):
    check_function(f, (Tensor(),))
    check_function(f, (Tensor(2),))
    check_function(f, (Tensor(2),), {'axis': Value(0)})
    check_function(f, (Tensor(2, 3),))
    check_function(f, (Tensor(2, 3),), {'axis': Value(0, 1)})


def test_logsumexp_correctness():
    mat = PositiveTensor(3, 4).np()
    allclose(B.logsumexp(mat, axis=1), scipy.special.logsumexp(mat, axis=1))


@pytest.mark.parametrize('f', [B.all, B.any])
def test_logical_reductions(f):
    check_function(f, (BoolTensor(),), {}, assert_dtype=False)
    check_function(f, (BoolTensor(2),), {}, assert_dtype=False)
    check_function(f, (BoolTensor(2),), {'axis': Value(0)}, assert_dtype=False)
    check_function(f, (BoolTensor(2, 3),), {}, assert_dtype=False)
    check_function(f, (BoolTensor(2, 3),), {'axis': Value(0, 1)},
                   assert_dtype=False)


@pytest.mark.parametrize('f', [B.lt, B.le, B.gt, B.ge])
def test_logical_comparisons(f):
    check_function(f, (Tensor(), Tensor()), {}, assert_dtype=False)
    check_function(f, (Tensor(2), Tensor(2)), {}, assert_dtype=False)
    check_function(f, (Tensor(2, 3), Tensor(2, 3)), {}, assert_dtype=False)


def test_scan():
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
    allclose(B.scan(scan_f, xs.np(), init_h.np(), init_y.np()),
             B.scan(scan_f, xs.tf(), init_h.tf(), init_y.tf()))

    # Check shape checking.

    def incorrect_scan_f(prev, x):
        return prev + prev

    with pytest.raises(RuntimeError):
        B.scan(incorrect_scan_f,
               Tensor(4).torch(),
               Tensor().torch(),
               Tensor().torch())


def test_sort():
    check_function(B.sort, (Tensor(4),),
                   {'axis': Value(-1, 0), 'descending': Bool()})
    # AutoGrad cannot sort multidimensional arrays.
    check_function(B.sort, (Tensor(2, 3, 4),),
                   {'axis': Value(-1, 0, 1, 2), 'descending': Bool()},
                   skip=[B.AGNumeric])


def test_argsort():
    check_function(B.argsort, (Tensor(4),),
                   {'axis': Value(-1, 0), 'descending': Bool()},
                   assert_dtype=False)
    # AutoGrad cannot sort multidimensional arrays.
    check_function(B.argsort, (Tensor(2, 3, 4),),
                   {'axis': Value(-1, 0, 1, 2), 'descending': Bool()},
                   skip=[B.AGNumeric],
                   assert_dtype=False)


def test_to_numpy():
    check_function(B.to_numpy, (Tensor(),))
    check_function(B.to_numpy, (Tensor(4),))


def test_to_numpy_multiple_objects():
    assert B.to_numpy(tf.constant(1), tf.constant(1)) == (1, 1)


def test_to_numpy_list():
    x = B.to_numpy([tf.constant(1)])
    assert isinstance(x[0], (B.Number, B.NPNumeric))


def test_to_numpy_tuple():
    x = B.to_numpy((tf.constant(1),))
    assert isinstance(x[0], (B.Number, B.NPNumeric))


def test_to_numpy_dict():
    x = B.to_numpy({'a': tf.constant(1)})
    assert isinstance(x['a'], (B.Number, B.NPNumeric))
