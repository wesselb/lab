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
    resolve_axis,
    as_tuple,
    abstract,
    batch_computation,
    _common_shape,
    _translate_index,
)

# noinspections PyUnresolvedReferences
from .util import approx, check_lazy_shapes


def test_resolve_axis(check_lazy_shapes):
    a = B.randn(2, 2, 2)

    # `None`s should just pass through.
    assert resolve_axis(a, None) is None

    # Test `negative = False`.
    with pytest.raises(ValueError):
        resolve_axis(a, -4)
    assert resolve_axis(a, -3) == 0
    assert resolve_axis(a, -2) == 1
    assert resolve_axis(a, -1) == 2
    assert resolve_axis(a, 0) == 0
    assert resolve_axis(a, 1) == 1
    assert resolve_axis(a, 2) == 2
    with pytest.raises(ValueError):
        resolve_axis(a, 3)

    # Test `negative = True`.
    with pytest.raises(ValueError):
        resolve_axis(a, -4, negative=True)
    assert resolve_axis(a, -3, negative=True) == -3
    assert resolve_axis(a, -2, negative=True) == -2
    assert resolve_axis(a, -1, negative=True) == -1
    assert resolve_axis(a, 0, negative=True) == -3
    assert resolve_axis(a, 1, negative=True) == -2
    assert resolve_axis(a, 2, negative=True) == -1
    with pytest.raises(ValueError):
        resolve_axis(a, 3, negative=True)


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
    approx(batch_computation(np.matmul, (x1, x2), (2, 2)), np.matmul(x1, x2))


def test_metadata(check_lazy_shapes):
    # Test that the name and docstrings for functions are available.
    assert B.transpose.__name__ == "transpose"
    assert B.transpose.__doc__ != ""


def test_abstract(check_lazy_shapes):
    # Test that `promote` and `promote_from` cannot be specified at the same time.
    with pytest.raises(ValueError):
        abstract(promote=1, promote_from=1)(lambda: None)

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
        pass

    @B.dispatch
    def f1(*args: Specific):
        return args

    @B.dispatch
    @abstract(promote=None)
    def f2(*args: General):
        pass

    @B.dispatch
    def f2(*args: Specific):
        return args

    @B.dispatch
    @abstract(promote=-1)
    def f3(*args: General):
        pass

    @B.dispatch
    def f3(*args: Specific):
        return args

    @B.dispatch
    @abstract(promote_from=-1)
    def f3_from(*args: General):
        pass

    @B.dispatch
    def f3_from(*args: Specific):
        return args

    @B.dispatch
    @abstract(promote=0)
    def f4(*args: General):
        pass

    @B.dispatch
    def f4(*args: Specific):
        return args

    @B.dispatch
    @abstract(promote_from=0)
    def f4_from(*args: General):
        pass

    @B.dispatch
    def f4_from(*args: Specific):
        return args

    @B.dispatch
    @abstract(promote=1)
    def f5(*args: General):
        pass

    @B.dispatch
    def f5(arg: Specific, *args: General):
        return (arg,) + args

    @B.dispatch
    @abstract(promote_from=1)
    def f5_from(*args: General):
        pass

    @B.dispatch
    def f5_from(arg: General, *args: Specific):
        return (arg,) + args

    @B.dispatch
    @abstract(promote=2)
    def f6(*args: General):
        pass

    @B.dispatch
    def f6(arg1: Specific, arg2: Specific, *args: General):
        return (arg1, arg2) + args

    @B.dispatch
    @abstract(promote_from=2)
    def f6_from(*args: General):
        pass

    @B.dispatch
    def f6_from(arg1: General, arg2: General, *args: Specific):
        return (arg1, arg2) + args

    # Register methods.
    B.f1 = f1
    B.f2 = f2
    B.f3 = f3
    B.f3_from = f3_from
    B.f4 = f4
    B.f4_from = f4_from
    B.f5 = f5
    B.f5_from = f5_from
    B.f6 = f6
    B.f6_from = f6_from

    # Test promotion.
    with pytest.raises(NotFoundLookupError):
        f1(a, a, a)

    with pytest.raises(NotFoundLookupError):
        f2(a, a, a)

    assert f3(a, a, a) == (b, b, b)
    with pytest.raises(NotFoundLookupError):
        f3_from(a, a, a)

    with pytest.raises(NotFoundLookupError):
        f4(a, a, a)
    assert f4_from(a, a, a) == (b, b, b)

    assert f5(a, a, a) == (b, a, a)
    assert f5(a) == (b,)
    assert f5_from(a, a, a) == (a, b, b)
    assert f5_from(a, a) == (a, b)

    assert f6(a, a, a) == (b, b, a)
    assert f6(a, a) == (b, b)
    assert f6_from(a, a, a, a) == (a, a, b, b)
    assert f6_from(a, a, a) == (a, a, b)

    # Put back promotion function.
    plum.promote = plum_promote
