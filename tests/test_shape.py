import pytest

import lab as B
from lab.shape import Shape, Dimension


def test_shape():
    shape = Shape(5, 2, 3)

    # Test indexing.
    assert shape[0] == 5
    assert shape[1] == 2
    assert shape[2] == 3
    assert isinstance(shape[0:1], Shape)
    assert shape[0:2] == Shape(5, 2)

    # Test comparisons.
    assert shape == Shape(5, 2, 3)
    assert shape != Shape(5, 2, 4)

    # Test concatenation with another shape.
    shape2 = Shape(7, 8, 9)
    assert shape + shape2 == Shape(5, 2, 3, 7, 8, 9)
    assert shape.__radd__(shape2) == Shape(7, 8, 9, 5, 2, 3)
    assert isinstance((shape + shape2).dims[0], int)
    assert isinstance((shape.__radd__(shape2)).dims[0], int)

    # Test concatenation with a tuple.
    assert shape + (7, 8, 9) == Shape(5, 2, 3, 7, 8, 9)
    assert (7, 8, 9) + shape == Shape(7, 8, 9, 5, 2, 3)
    assert isinstance((shape + (7, 8, 9)).dims[0], int)
    assert isinstance(((7, 8, 9) + shape).dims[0], int)

    # Test conversion of doubly wrapped indices.
    assert isinstance(Shape(Dimension(1)).dims[0], int)

    # Test other operations.
    assert reversed(shape) == Shape(3, 2, 5)
    assert len(shape) == 3
    assert tuple(shape) == (Dimension(5), Dimension(2), Dimension(3))

    # Test representation.
    assert str(Shape()) == "()"
    assert repr(Shape()) == "Shape()"
    assert str(Shape(1)) == "(1,)"
    assert repr(Shape(1)) == "Shape(1)"
    assert str(Shape(1, 2)) == "(1, 2)"
    assert repr(Shape(1, 2)) == "Shape(1, 2)"

    # Test conversion to NumPy.
    assert isinstance(B.to_numpy(Shape(1, 2)), tuple)
    assert B.to_numpy(Shape(1, 2)) == (1, 2)


def test_dimension():
    d = Dimension(5)

    assert int(d) is 5
    with pytest.raises(TypeError) as e:
        len(d)
    assert "object of type 'int' has no len()" in str(e.value)
    with pytest.raises(TypeError) as e:
        iter(d)
    assert "'int' object is not iterable" in str(e.value)

    # Test comparisons.
    assert d == 5
    assert d >= 5
    assert d > 4
    assert d <= 5
    assert d < 6

    # Test that the dimension automatically unwraps.
    assert d + 1 is 6
    assert 1 + d is 6
    assert d - 1 is 4
    assert 1 - d is -4
    assert d * 1 is 5
    assert 1 * d is 5
    assert isinstance(d / 5, float)
    assert d / 5 == 1
    assert d ** 2 is 25

    # Test representation.
    assert repr(d) == str(d) == "5"
