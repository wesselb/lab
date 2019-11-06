import numpy as np
import pytest
import tensorflow as tf

import lab as B
from .util import (
    check_function,
    Tensor,
    Matrix,
    Value,
    List,
    Tuple,
    allclose
)


@pytest.mark.parametrize('f', [B.shape, B.rank, B.length, B.size])
def test_sizing(f):
    check_function(f, (Tensor(),), {}, assert_dtype=False)
    check_function(f, (Tensor(3, ),), {}, assert_dtype=False)
    check_function(f, (Tensor(3, 4),), {}, assert_dtype=False)
    check_function(f, (Tensor(3, 4, 5),), {}, assert_dtype=False)


@pytest.mark.parametrize('x,shape', [([], (0,)),
                                     ([5], (1,)),
                                     ([[5], [6]], (2, 1)),
                                     ((), (0,)),
                                     ((5,), (1,)),
                                     (((5,), (2,)), (2, 1))])
def test_shape_other(x, shape):
    assert B.shape(x) == shape


def test_isscalar():
    assert B.isscalar(1.0)
    assert not B.isscalar(np.array([1.0]))


def test_expand_dims():
    check_function(B.expand_dims, (Tensor(3, 4, 5),), {'axis': Value(0, 1)})


def test_squeeze():
    check_function(B.squeeze, (Tensor(3, 4, 5),))
    check_function(B.squeeze, (Tensor(1, 4, 5),))
    check_function(B.squeeze, (Tensor(3, 1, 5),))
    check_function(B.squeeze, (Tensor(1, 4, 1),))


def test_uprank():
    allclose(B.uprank(1.0), np.array([[1.0]]))
    allclose(B.uprank(np.array([1.0, 2.0])), np.array([[1.0], [2.0]]))
    allclose(B.uprank(np.array([[1.0, 2.0]])), np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError):
        B.uprank(np.array([[[1.0]]]))


def test_diag():
    check_function(B.diag, (Tensor(3),))
    check_function(B.diag, (Tensor(3, 3),))
    with pytest.raises(ValueError):
        B.diag(Tensor().tf())


def test_flatten():
    check_function(B.flatten, (Tensor(3),))
    check_function(B.flatten, (Tensor(3, 4),))


def test_vec_to_tril_and_back():
    check_function(B.vec_to_tril, (Tensor(6),))
    check_function(B.tril_to_vec, (Matrix(3),))

    # Check correctness.
    mat = Tensor(6).np()
    allclose(B.tril_to_vec(B.vec_to_tril(mat)), mat)

    # Check exceptions.
    for x in Matrix(3, 4).forms():
        with pytest.raises(ValueError):
            B.vec_to_tril(x)
        with pytest.raises(ValueError):
            B.tril_to_vec(x)
    for x in Matrix(3, 4, 5).forms():
        with pytest.raises(ValueError):
            B.tril_to_vec(x)


def test_stack():
    check_function(B.stack, (Matrix(3), Matrix(3), Matrix(3)),
                   {'axis': Value(0, 1)})


def test_unstack():
    check_function(B.unstack, (Tensor(3, 4, 5),), {'axis': Value(0, 1, 2)})


def test_reshape():
    check_function(B.reshape, (Tensor(3, 4, 5), Value(3), Value(20)))
    check_function(B.reshape, (Tensor(3, 4, 5), Value(12), Value(5)))


def test_concat():
    check_function(B.concat, (Matrix(3), Matrix(3), Matrix(3)),
                   {'axis': Value(0, 1)})


def test_concat2d():
    check_function(B.concat2d,
                   (List(Matrix(3), Matrix(3)), List(Matrix(3), Matrix(3))))
    check_function(B.concat2d,
                   (Tuple(Matrix(3), Matrix(3)), Tuple(Matrix(3), Matrix(3))))


@pytest.mark.parametrize('r1', [1, 2])
@pytest.mark.parametrize('r2', [1, 2])
def test_tile(r1, r2):
    check_function(B.tile, (Tensor(3, 4), Value(r1), Value(r2)))


def test_take_consistency():
    # Check consistency between indices and mask.
    check_function(B.take, (Matrix(3, 3), Value([0, 1], [True, True, False])),
                   {'axis': Value(0, 1)})


def test_take_consistency_order():
    # Check order of indices.
    check_function(B.take,
                   (Matrix(3, 4), Value([2, 1])),
                   {'axis': Value(0, 1)})


def test_take_indices_rank():
    # Check that indices must be rank 1.
    for a in Matrix(3, 4).forms():
        with pytest.raises(ValueError):
            B.take(a, [[0], [1]])


def test_take_empty_list():
    # Check empty list.
    check_function(B.take, (Matrix(3, 4), Value([])), {'axis': Value(0, 1)})


def test_take_tf():
    # Check that TensorFlow also takes in tensors.
    a = Matrix(3, 4, 5)
    ref = Tensor(3)
    allclose(B.take(a.tf(), ref.tf() > 0), B.take(a.np(), ref.np() > 0))
    allclose(B.take(a.tf(), B.range(tf.int64, 2)),
             B.take(a.np(), B.range(2)))
