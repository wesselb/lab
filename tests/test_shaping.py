import numpy as np
import pytest
import tensorflow as tf
from plum import NotFoundLookupError

import lab as B
from lab.shape import Shape

# noinspection PyUnresolvedReferences
from .util import (
    check_function,
    Tensor,
    Matrix,
    Value,
    List,
    Tuple,
    approx,
    check_lazy_shapes,
)


@pytest.mark.parametrize("f", [B.shape, B.rank, B.length, B.size])
def test_sizing(f, check_lazy_shapes):
    check_function(f, (Tensor(),), {}, assert_dtype=False)
    check_function(
        f,
        (
            Tensor(
                3,
            ),
        ),
        {},
        assert_dtype=False,
    )
    check_function(f, (Tensor(3, 4),), {}, assert_dtype=False)
    check_function(f, (Tensor(3, 4, 5),), {}, assert_dtype=False)


@pytest.mark.parametrize(
    "x,shape",
    [
        (1, ()),
        ([], (0,)),
        ([5], (1,)),
        ([[5], [6]], (2, 1)),
        ((), (0,)),
        ((5,), (1,)),
        (((5,), (2,)), (2, 1)),
    ],
)
def test_shape(x, shape, check_lazy_shapes):
    assert B.shape(x) == shape


def test_subshape(check_lazy_shapes):
    assert B.shape(B.zeros(2), 0) == 2
    assert B.shape(B.zeros(2, 3, 4), 1) == 3
    assert B.shape(B.zeros(2, 3, 4), 0, 2) == (2, 4)
    assert B.shape(B.zeros(2, 3, 4), 0, 1, 2) == (2, 3, 4)

    # Check for possible infinite recursion.
    with pytest.raises(NotFoundLookupError):
        B.shape(None, 1)


def test_lazy_shape():
    a = B.randn(2, 2)

    # By default, it should be off.
    assert isinstance(B.shape(a), tuple)

    # Turn on.
    with B.lazy_shapes():
        assert isinstance(B.shape(a), Shape)

        # Force lazy shapes to be off again.
        B.lazy_shapes.enabled = False
        assert isinstance(B.shape(a), tuple)

        # Turn on again.
        with B.lazy_shapes():
            assert isinstance(B.shape(a), Shape)

        # Should remain off.
        assert isinstance(B.shape(a), tuple)


def test_isscalar(check_lazy_shapes):
    assert B.isscalar(1.0)
    assert not B.isscalar(np.array([1.0]))


def test_expand_dims(check_lazy_shapes):
    check_function(B.expand_dims, (Tensor(3, 4, 5),), {"axis": Value(0, 1)})


def test_squeeze(check_lazy_shapes):
    check_function(B.squeeze, (Tensor(3, 4, 5),))
    check_function(B.squeeze, (Tensor(1, 4, 5),))
    check_function(B.squeeze, (Tensor(3, 1, 5),))
    check_function(B.squeeze, (Tensor(1, 4, 1),))

    # Test squeezing lists and tuples
    assert B.squeeze((1,)) == 1
    assert B.squeeze((1, 2)) == (1, 2)
    assert B.squeeze([1]) == 1
    assert B.squeeze([1, 2]) == [1, 2]


def test_uprank(check_lazy_shapes):
    # `rank=2`, the default:
    approx(B.uprank(1.0), np.array([[1.0]]))
    approx(B.uprank(np.array([1.0, 2.0])), np.array([[1.0], [2.0]]))
    approx(B.uprank(np.array([[1.0, 2.0]])), np.array([[1.0, 2.0]]))
    approx(B.uprank(np.array([[[1.0]]])), np.array([[[1.0]]]))

    # `rank=1`:
    approx(B.uprank(1.0, rank=1), np.array([1.0]))
    approx(B.uprank(np.array([1.0, 2.0]), rank=1), np.array([1.0, 2.0]))
    approx(B.uprank(np.array([[1.0, 2.0]]), rank=1), np.array([[1.0, 2.0]]))


@pytest.mark.parametrize("source_shape", [(1, 1, 1), (1, 1, 4), (1, 3, 4), (2, 3, 4)])
def test_broadcast_to(check_lazy_shapes, source_shape):
    def f(x):
        return B.broadcast_to(x, 2, 3, 4)

    check_function(f, (Tensor(*source_shape),))


def test_diag(check_lazy_shapes):
    check_function(B.diag, (Tensor(3),))
    check_function(B.diag, (Tensor(3, 3),))
    # Test rank check for TensorFlow.
    with pytest.raises(ValueError):
        B.diag(Tensor().tf())


def test_diag_extract(check_lazy_shapes):
    check_function(B.diag_extract, (Tensor(3, 3),))
    check_function(B.diag_extract, (Tensor(2, 3, 3),))


def test_diag_construct(check_lazy_shapes):
    check_function(B.diag_construct, (Tensor(3),))
    check_function(B.diag_construct, (Tensor(2, 3),))
    # Test rank check for fallback.
    with pytest.raises(ValueError):
        B.diag_construct(Tensor().np())


def test_flatten(check_lazy_shapes):
    check_function(B.flatten, (Tensor(3),))
    check_function(B.flatten, (Tensor(3, 4),))


@pytest.mark.parametrize("offset", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("batch_shape", [(), (5,)])
def test_vec_to_tril(offset, batch_shape, check_lazy_shapes):
    n = B.length(B.tril_to_vec(B.ones(7, 7), offset=offset))
    check_function(B.vec_to_tril, (Tensor(*batch_shape, n),), {"offset": Value(offset)})


@pytest.mark.parametrize("batch_shape", [(), (5,)])
def test_tril_to_vec(batch_shape, check_lazy_shapes):
    check_function(
        B.tril_to_vec, (Tensor(*batch_shape, 6, 6),), {"offset": Value(-1, 0, 1)}
    )


@pytest.mark.parametrize("offset", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("batch_shape", [(), (5,)])
def test_vec_to_tril_and_back_correctness(offset, batch_shape, check_lazy_shapes):
    n = B.length(B.tril_to_vec(B.ones(7, 7), offset=offset))
    for vec in Tensor(*batch_shape, n).forms():
        mat = B.vec_to_tril(vec, offset=offset)
        approx(B.tril_to_vec(mat, offset=offset), vec)


def test_vec_to_tril_and_back_exceptions(check_lazy_shapes):
    # Check rank checks.
    for x in Tensor().forms():
        with pytest.raises(ValueError):
            B.vec_to_tril(x)
        with pytest.raises(ValueError):
            B.tril_to_vec(x)
    for x in Tensor(3).forms():
        with pytest.raises(ValueError):
            B.tril_to_vec(x)

    # Check square checks.
    for x in Tensor(3, 4).forms():
        with pytest.raises(ValueError):
            B.tril_to_vec(x)
    for x in Tensor(3, 4, 5).forms():
        with pytest.raises(ValueError):
            B.tril_to_vec(x)


def test_stack(check_lazy_shapes):
    check_function(B.stack, (Matrix(3), Matrix(3), Matrix(3)), {"axis": Value(0, 1)})


def test_unstack(check_lazy_shapes):
    check_function(B.unstack, (Tensor(3, 4, 5),), {"axis": Value(0, 1, 2)})


def test_reshape(check_lazy_shapes):
    check_function(B.reshape, (Tensor(3, 4, 5), Value(3), Value(20)))
    check_function(B.reshape, (Tensor(3, 4, 5), Value(12), Value(5)))


def test_concat(check_lazy_shapes):
    check_function(B.concat, (Matrix(3), Matrix(3), Matrix(3)), {"axis": Value(0, 1)})


def test_concat2d(check_lazy_shapes):
    check_function(B.concat2d, (List(Matrix(3), Matrix(3)), List(Matrix(3), Matrix(3))))


@pytest.mark.parametrize("r1", [1, 2])
@pytest.mark.parametrize("r2", [1, 2])
def test_tile(r1, r2, check_lazy_shapes):
    check_function(B.tile, (Tensor(3, 4), Value(r1), Value(r2)))


def test_take_consistency(check_lazy_shapes):
    # Check consistency between indices and mask.
    check_function(
        B.take,
        (Matrix(3, 3), Value([0, 1], [True, True, False])),
        {"axis": Value(0, 1)},
    )


def test_take_consistency_order(check_lazy_shapes):
    # Check order of indices.
    check_function(B.take, (Matrix(3, 4), Value([2, 1])), {"axis": Value(0, 1)})


def test_take_indices_rank(check_lazy_shapes):
    # Check that indices must be rank 1.
    for a in Matrix(3, 4).forms():
        with pytest.raises(ValueError):
            B.take(a, [[0], [1]])


def test_take_empty_list(check_lazy_shapes):
    # Check empty list.
    check_function(B.take, (Matrix(3, 4), Value([])), {"axis": Value(0, 1)})


def test_take_tf(check_lazy_shapes):
    # Check that TensorFlow also takes in tensors.
    a = Matrix(3, 4, 5)
    ref = Tensor(3)
    approx(B.take(a.tf(), ref.tf() > 0), B.take(a.np(), ref.np() > 0))
    approx(B.take(a.tf(), B.range(tf.int64, 2)), B.take(a.np(), B.range(2)))
