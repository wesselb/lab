from plum import Dispatcher
from functools import wraps

__all__ = ["Shape", "Dimension", "dispatch_unwrap_dimensions"]

_dispatch = Dispatcher()


class Shape:
    """A shape.

    Args:
        *dims (number): Dimensions of the shape.

    Attributes:
        dims (tuple[number]): Dimensions of the shape.
    """

    def __init__(self, *dims):
        self.dims = dims

    def __getitem__(self, item):
        return Dimension(self.dims[item])

    def __len__(self):
        return len(self.dims)

    def __iter__(self):
        for dim in self.dims:
            yield Dimension(dim)

    def __eq__(self, other):
        return len(self) == len(other) and all(x == y for x, y in zip(self, other))

    def __reversed__(self):
        return Shape(*reversed(self.dims))

    def __repr__(self):
        return "Shape(" + ", ".join(repr(x) for x in self) + ")"

    def __str__(self):
        return "Shape(" + ", ".join(str(x) for x in self) + ")"


class Dimension:
    """A dimension in a shape.

    Args:
        dim (number): Dimension.

    Attributes:
        dim (number): Dimension.
    """

    def __init__(self, dim):
        self.dim = dim

    def __int__(self):
        return int(self.dim)

    def __len__(self):
        return len(self.dim)

    def __iter__(self):
        return iter(self.dim)

    def __eq__(self, other):
        return self.dim == other

    def __ge__(self, other):
        return self.dim >= other

    def __gt__(self, other):
        return self.dim > other

    def __le__(self, other):
        return self.dim <= other

    def __lt__(self, other):
        return self.dim < other

    def __add__(self, other):
        return self.dim + other

    def __radd__(self, other):
        return other + self.dim

    def __sub__(self, other):
        return self.dim - other

    def __rsub__(self, other):
        return other - self.dim

    def __mul__(self, other):
        return self.dim * other

    def __rmul__(self, other):
        return other * self.dim

    def __truediv__(self, other):
        return self.dim / other

    def __pow__(self, power, modulo=None):
        return self.dim.__pow__(power, modulo)

    def __repr__(self):
        return repr(self.dim)

    def __str__(self):
        return str(self.dim)


@_dispatch(object)
def unwrap_dimension(a):
    """Unwrap a dimension.

    Args:
        a (object): Dimension to unwrap.

    Returns:
        number: If `a` was wrapped with :class:`.shape.Dimension`, then this will be
            `a.dim`. Otherwise, the result is just `a`.
    """
    return a


@_dispatch(Dimension)
def unwrap_dimension(a):
    return a.dim


def dispatch_unwrap_dimensions(dispatch):
    """Unwrap all dimensions after performing dispatch.

    Args:
        dispatch (decorator): Dispatch decorator.
    """

    def unwrapped_dispatch(*dispatch_args, **dispatch_kw_args):
        def decorator(f):
            @wraps(f)
            def f_wrapped(*args, **kw_args):
                return f(*(unwrap_dimension(arg) for arg in args), **kw_args)

            return dispatch(*dispatch_args, **dispatch_kw_args)(f_wrapped)

        return decorator

    return unwrapped_dispatch
