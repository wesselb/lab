# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from . import dispatch
from .types import Numeric, Dimension
from .util import abstract

__all__ = ['shape',
           'shape_int',
           'rank',
           'length', 'size',
           'isscalar',
           'expand_dims',
           'squeeze',
           'uprank',
           'diag',
           'flatten',
           'vec_to_tril',
           'tril_to_vec',
           'stack',
           'unstack',
           'reshape',
           'concat',
           'concat2d',
           'take']


@dispatch(Numeric)
@abstract()
def shape(a):  # pragma: no cover
    """Get the shape of a tensor.

    Args:
        a (tensor): Tensor.

    Returns:
        object: Shape of `a`.
    """


@dispatch(Numeric)
@abstract()
def shape_int(a):  # pragma: no cover
    """Get the shape of a tensor as a tuple of integers.

    Args:
        a (tensor): Tensor.

    Returns:
        tuple: Shape of `a` as a tuple of integers.
    """


@dispatch(Numeric)
@abstract()
def rank(a):  # pragma: no cover
    """Get the shape of a tensor.

    Args:
        a (tensor): Tensor.

    Returns:
        int: Rank of `a`.
    """


@dispatch(Numeric)
@abstract()
def length(a):  # pragma: no cover
    """Get the length of a tensor.

    Args:
        a (tensor): Tensor.

    Returns:
        int: Length of `a`.
    """


size = length  #: Alias for `length`.


@dispatch(Numeric)
def isscalar(a):
    """Check whether a tensor is a scalar.

    Args:
        a (tensor): Tensor.

    Returns:
        bool: `True` if `a` is a scalar, otherwise `False`.
    """
    return rank(a) == 0


@dispatch(Numeric)
@abstract()
def expand_dims(a, axis=0):  # pragma: no cover
    """Insert an empty axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Index of new axis. Defaults to `0`.

    Returns:
        tensor: `a` with the new axis.
    """


@dispatch(Numeric)
@abstract()
def squeeze(a):  # pragma: no cover
    """Remove all axes containing only a single element.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: `a` without axes containing only a single element.
    """


@dispatch(Numeric)
def uprank(a):
    """Convert the input into a rank two tensor.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: `a`, but of rank two.
    """
    a_rank = rank(a)
    if a_rank > 2:
        raise ValueError('Cannot convert a tensor of rank {} to rank 2.'
                         ''.format(a_rank))
    while a_rank < 2:
        a = expand_dims(a, axis=-1)
        a_rank += 1
    return a


@dispatch(Numeric)
@abstract()
def diag(a):  # pragma: no cover
    """Take the diagonal of a matrix, or construct a diagonal matrix from its
    diagonal.

    Args:
        a (tensor): Matrix or diagonal.

    Returns:
        tensor: Diagonal or matrix.
    """


@dispatch(Numeric)
def flatten(a):  # pragma: no cover
    """Flatten a tensor.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Flattened tensor.
    """
    return reshape(a, -1)


def _vec_to_tril_shape(a):
    n = shape_int(a)[0]
    return int(((1 + 8 * n) ** .5 - 1) / 2)


@dispatch(Numeric)
@abstract()
def vec_to_tril(a):  # pragma: no cover
    """Construct a lower triangular matrix from a vector.

    Args:
        a (tensor): Vector.

    Returns:
        tensor: Lower triangular matrix.
    """


@dispatch(Numeric)
@abstract()
def tril_to_vec(a):  # pragma: no cover
    """Construct a vector from a lower triangular matrix.

    Args:
        a (tensor): Lower triangular matrix.

    Returns:
        tensor: Vector
    """


@dispatch([Numeric])
@abstract()
def stack(*elements, **kw_args):  # pragma: no cover
    """Concatenate tensors along a new axis.

    Args:
        *elements (tensor): Tensors to stack.
        axis (int, optional): Index of new axis. Defaults to `0`.

    Returns:
        tensor: Stacked tensors.
    """


@dispatch(Numeric)
@abstract()
def unstack(a, axis=0):  # pragma: no cover
    """Unstack tensors along an axis.

    Args:
        a (list): List of tensors.
        axis (int, optional): Index along which to unstack. Defaults to `0`.

    Returns:
        list[tensor]: List of tensors.
    """


@dispatch(Numeric, [Dimension])
@abstract()
def reshape(a, *shape):  # pragma: no cover
    """Reshape a tensor.

    Args:
        a (tensor): Tensor to reshape.
        *shape (shape): New shape.

    Returns:
        tensor: Reshaped tensor.
    """


@dispatch([Numeric])
@abstract()
def concat(*elements, **kw_args):  # pragma: no cover
    """Concatenate tensors along an axis.

    Args:
        *elements (tensor): Tensors to concatenate
        axis (int, optional): Axis along which to concatenate. Defaults to `0`.

    Returns:
        tensor: Concatenation.
    """


@dispatch([{list, tuple}])
def concat2d(*rows):
    """Concatenate tensors into a matrix.

    Args:
        *rows (list[list[tensor]]): List of list of tensors, which form the
            rows of the matrix.

    Returns:
        tensor: Assembled matrix.
    """
    return concat(*[concat(*row, axis=1) for row in rows], axis=0)


@dispatch(Numeric, object)
@abstract(promote=None)
def take(a, indices, axis=0):  # pragma: no cover
    """Take particular elements along an axis.

    Args:
        a (tensor): Tensor.
        indices (list): List of indices to take.
        axis (int, optional): Axis along which to take indices. Defaults to `0`.

    Returns:
        tensor: Selected subtensor.
    """