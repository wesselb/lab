import pytest
from lab.shape import Shape, Dimension


def test_shape():
    shape = Shape(5, 2, 3)

    assert shape[0] == 5
    assert shape[1] == 2
    assert shape[2] == 3
    assert len(shape) == 3
    assert tuple(shape) == (Dimension(5), Dimension(2), Dimension(3))
    assert shape == Shape(5, 2, 3)
    assert shape != Shape(5, 2, 4)
    assert reversed(shape) == Shape(3, 2, 5)
    assert repr(shape) == str(shape) == "Shape(5, 2, 3)"


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
    assert repr(d) == str(d) == "5"
