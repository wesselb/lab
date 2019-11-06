import numpy as np
import plum
import pytest
from plum import NotFoundLookupError

import lab as B
import lab.autograd as B_autograd
import lab.tensorflow as B_tf
import lab.torch as B_torch
from lab.util import (
    abstract,
    batch_computation,
    _common_shape,
    _translate_index
)
from .util import allclose


@pytest.mark.parametrize('other', [B_tf, B_torch, B_autograd])
def test_module_mapping(other):
    assert B is other


@pytest.mark.parametrize('shapes,common_shape',
                         [([(), ()], ()),
                          ([(5,), (1,)], (5,)),
                          ([(2, 5), (1, 5)], (2, 5)),
                          ([(5,), (1, 5)], (1, 5)),
                          ([(3, 5), (1,)], (3, 5))])
def test_common_shape(shapes, common_shape):
    assert _common_shape(*shapes) == common_shape
    assert _common_shape(*reversed(shapes)) == common_shape


@pytest.mark.parametrize('shapes',
                         [[(5,), (6,)],
                          [(5, 2), (5, 3)],
                          [(5, 2), (3,)]])
def test_common_shape_errors(shapes):
    with pytest.raises(RuntimeError):
        _common_shape(*shapes)
    with pytest.raises(RuntimeError):
        _common_shape(*reversed(shapes))


@pytest.mark.parametrize('index,batch_shape,translated_index',
                         [((5, 2), (3,), (2,)),
                          ((2, 3, 4), (5, 5), (3, 4)),
                          ((2, 3, 4), (1, 5), (0, 4)),
                          ((2, 3, 4), (5, 1), (3, 0))])
def test_translate_index(index, batch_shape, translated_index):
    assert _translate_index(index, batch_shape) == translated_index


@pytest.mark.parametrize('index,batch_shape',
                         [((5, 3), (3,)),
                          ((2, 3, 4), (4, 4))])
def test_translate_index_errors(index, batch_shape):
    with pytest.raises(RuntimeError):
        _translate_index(index, batch_shape)


@pytest.mark.parametrize('x1_batch', [(), (1,), (2,), (2, 2), (2, 1), (1, 2)])
@pytest.mark.parametrize('x2_batch', [(), (1,), (2,), (2, 2), (2, 1), (1, 2)])
def test_batch_computation(x1_batch, x2_batch):
    x1 = np.random.randn(*(x1_batch + (3, 4)))
    x2 = np.random.randn(*(x2_batch + (4, 5)))
    allclose(batch_computation(np.matmul, (x1, x2), (2, 2)),
             np.matmul(x1, x2))


def test_metadata():
    # Test that the name and docstrings for functions are available.
    assert B.transpose.__name__ == 'transpose'
    assert B.transpose.__doc__ != ''


def test_abstract():
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
    @B.dispatch([General])
    @abstract()
    def f1(*args):
        return args

    @B.dispatch([Specific])
    def f1(*args):
        return args

    @B.dispatch([General])
    @abstract(promote=None)
    def f2(*args):
        return args

    @B.dispatch([Specific])
    def f2(*args):
        return args

    @B.dispatch([General])
    @abstract(promote=-1)
    def f3(*args):
        return args

    @B.dispatch([Specific])
    def f3(*args):
        return args

    @B.dispatch([General])
    @abstract(promote=0)
    def f4(*args):
        return args

    @B.dispatch([Specific])
    def f4(*args):
        return args

    @B.dispatch([General])
    @abstract(promote=1)
    def f5(*args):
        return args

    @B.dispatch(Specific, [General])
    def f5(*args):
        return args

    @B.dispatch([General])
    @abstract(promote=2)
    def f6(*args):
        return args

    @B.dispatch(Specific, Specific, [General])
    def f6(*args):
        return args

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
