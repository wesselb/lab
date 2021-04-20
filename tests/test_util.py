import numpy as np
import plum
import pytest
from plum import NotFoundLookupError

import lab as B
import lab.autograd as B_autograd
import lab.tensorflow as B_tf
import lab.torch as B_torch
import lab.jax as B_jax
from lab.util import (
    as_tuple,
    abstract,
    batch_computation,
    _common_shape,
    _translate_index,
)

# noinspections PyUnresolvedReferences
from .util import allclose, check_lazy_shapes


@pytest.mark.parametrize("other", [B_tf, B_torch, B_autograd, B_jax])
def test_module_mapping(other, check_lazy_shapes):
    assert B is other


def test_as_tuple(check_lazy_shapes):
    assert as_tuple(1) == (1,)
    assert as_tuple((1,)) == (1,)
    assert as_tuple((1, 2)) == (1, 2)


@pytest.mark.parametrize(
    "shapes,common_shape",
    [
        ([(), ()], ()),
        ([(5,), (1,)], (5,)),
        ([(2, 5), (1, 5)], (2, 5)),
        ([(5,), (1, 5)], (1, 5)),
        ([(3, 5), (1,)], (3, 5)),
    ],
)
def test_common_shape(shapes, common_shape, check_lazy_shapes):
    assert _common_shape(*shapes) == common_shape
    assert _common_shape(*reversed(shapes)) == common_shape


@pytest.mark.parametrize("shapes", [[(5,), (6,)], [(5, 2), (5, 3)], [(5, 2), (3,)]])
def test_common_shape_errors(shapes, check_lazy_shapes):
    with pytest.raises(RuntimeError):
        _common_shape(*shapes)
    with pytest.raises(RuntimeError):
        _common_shape(*reversed(shapes))


@pytest.mark.parametrize(
    "index,batch_shape,translated_index",
    [
        ((5, 2), (3,), (2,)),
        ((2, 3, 4), (5, 5), (3, 4)),
        ((2, 3, 4), (1, 5), (0, 4)),
        ((2, 3, 4), (5, 1), (3, 0)),
    ],
)
def test_translate_index(index, batch_shape, translated_index, check_lazy_shapes):
    assert _translate_index(index, batch_shape) == translated_index


@pytest.mark.parametrize("index,batch_shape", [((5, 3), (3,)), ((2, 3, 4), (4, 4))])
def test_translate_index_errors(index, batch_shape, check_lazy_shapes):
    with pytest.raises(RuntimeError):
        _translate_index(index, batch_shape)


@pytest.mark.parametrize("x1_batch", [(), (1,), (2,), (2, 2), (2, 1), (1, 2)])
@pytest.mark.parametrize("x2_batch", [(), (1,), (2,), (2, 2), (2, 1), (1, 2)])
def test_batch_computation(x1_batch, x2_batch, check_lazy_shapes):
    x1 = np.random.randn(*(x1_batch + (3, 4)))
    x2 = np.random.randn(*(x2_batch + (4, 5)))
    allclose(batch_computation(np.matmul, (x1, x2), (2, 2)), np.matmul(x1, x2))


def test_metadata(check_lazy_shapes):
    # Test that the name and docstrings for functions are available.
    assert B.transpose.__name__ == "transpose"
    assert B.transpose.__doc__ != ""


def test_abstract(check_lazy_shapes):
    class General:
        pass

    class Specific:
        pass

    a = General()
    b = Specific()

    # Temporarily mock Plum's promotion function.
    plum_promote = plum.promote
    plum.promote = lambda *args: (b,) * len(args)

    # Define some abstract functions.
    @B.dispatch
    @abstract()
    def f1(*args: General):
        return args

    @B.dispatch
    def f1(*args: Specific):
        return args

    @B.dispatch
    @abstract(promote=None)
    def f2(*args: General):
        return args

    @B.dispatch
    def f2(*args: Specific):
        return args

    @B.dispatch
    @abstract(promote=-1)
    def f3(*args: General):
        return args

    @B.dispatch
    def f3(*args: Specific):
        return args

    @B.dispatch
    @abstract(promote=0)
    def f4(*args: General):
        return args

    @B.dispatch
    def f4(*args: Specific):
        return args

    @B.dispatch
    @abstract(promote=1)
    def f5(*args: General):
        return args

    @B.dispatch
    def f5(arg: Specific, *args: General):
        return (arg,) + args

    @B.dispatch
    @abstract(promote=2)
    def f6(*args: General):
        return args

    @B.dispatch
    def f6(arg1: Specific, arg2: Specific, *args: General):
        return (arg1, arg2) + args

    # Register methods.
    B.f1 = f1
    B.f2 = f2
    B.f3 = f3
    B.f4 = f4
    B.f5 = f5
    B.f6 = f6

    # Test promotion.
    assert f1(a, a, a) == (b, b, b)
    with pytest.raises(NotFoundLookupError):
        f2(a, a, a)
    assert f3(a, a, a) == (b, b, b)
    with pytest.raises(NotFoundLookupError):
        f4(a, a, a)
    assert f5(a, a, a) == (b, a, a)
    assert f5(a) == (b,)
    assert f6(a, a, a) == (b, b, a)
    assert f6(a, a) == (b, b)

    # Put back promotion function.
    plum.promote = plum_promote
