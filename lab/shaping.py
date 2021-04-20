import math

import numpy as np
from plum import Union

from . import B, dispatch
from .shape import Shape
from .types import Numeric, Int
from .util import abstract

__all__ = [
    "lazy_shapes",
    "shape",
    "rank",
    "length",
    "size",
    "isscalar",
    "expand_dims",
    "squeeze",
    "uprank",
    "diag",
    "diag_extract",
    "diag_construct",
    "flatten",
    "vec_to_tril",
    "tril_to_vec",
    "stack",
    "unstack",
    "reshape",
    "concat",
    "concat2d",
    "tile",
    "take",
]


class LazyShapes:
    """Simple context manager that tracks the status for lazy shapes.

    Attributes:
        enabled (bool): Are lazy shapes enabled?
    """

    def __init__(self):
        self.enabled = False

    def __enter__(self):
        self.enabled = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.enabled = False


lazy_shapes = LazyShapes()  #: Enable lazy shapes.


@dispatch
def shape(a: Numeric):  # pragma: no cover
    """Get the shape of a tensor.

    Args:
        a (tensor): Tensor.

    Returns:
        object: Shape of `a`.
    """
    try:
        if lazy_shapes.enabled:
            return Shape(*a.shape)
        else:
            return a.shape
    except AttributeError:
        # `a` must be a number.
        return ()


@dispatch
def shape(a: Union[list, tuple]):
    return np.array(a).shape


@dispatch
def rank(a: Union[Numeric, list, tuple]):  # pragma: no cover
    """Get the shape of a tensor.

    Args:
        a (tensor): Tensor.

    Returns:
        int: Rank of `a`.
    """
    return len(shape(a))


@dispatch
@abstract()
def length(a: Numeric):  # pragma: no cover
    """Get the length of a tensor.

    Args:
        a (tensor): Tensor.

    Returns:
        int: Length of `a`.
    """


size = length  #: Alias for `length`.


@dispatch
def isscalar(a: Numeric):
    """Check whether a tensor is a scalar.

    Args:
        a (tensor): Tensor.

    Returns:
        bool: `True` if `a` is a scalar, otherwise `False`.
    """
    return rank(a) == 0


@dispatch
@abstract()
def expand_dims(a: Numeric, axis=0):  # pragma: no cover
    """Insert an empty axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Index of new axis. Defaults to `0`.

    Returns:
        tensor: `a` with the new axis.
    """


@dispatch
@abstract()
def squeeze(a: Numeric):  # pragma: no cover
    """Remove all axes containing only a single element.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: `a` without axes containing only a single element.
    """


@dispatch
def squeeze(a: Union[tuple, list]):
    if len(a) == 1:
        return a[0]
    else:
        return a


@dispatch
def uprank(a: Numeric, rank=2):
    """Convert the input into a tensor of at least rank `rank`.

    Args:
        a (tensor): Tensor.
        rank (int, optional): Rank. Defaults to `2`.

    Returns:
        tensor: `a`, but of rank two.
    """
    a_rank = B.rank(a)
    while a_rank < rank:
        a = expand_dims(a, axis=-1)
        a_rank += 1
    return a


@dispatch
@abstract()
def diag(a: Numeric):
    """Take the diagonal of a matrix, or construct a diagonal matrix from its
    diagonal.

    Args:
        a (tensor): Matrix or diagonal.

    Returns:
        tensor: Diagonal or matrix.
    """


@dispatch
@abstract()
def diag_extract(a: Numeric):  # pragma: no cover
    """Take the diagonal of a matrix.

    Args:
        a (tensor): Matrix.

    Returns:
        tensor: Diagonal of matrix.
    """


@dispatch
def diag_construct(a: Numeric):  # pragma: no cover
    """Construct a diagonal matrix from its diagonal.

    Args:
        a (tensor): Diagonal.

    Returns:
        tensor: Matrix.
    """
    if B.rank(a) == 0:
        raise ValueError("Input must have at least rank 1.")

    # The one-dimensional case is better handled by `diag`.
    if B.rank(a) == 1:
        return B.diag(a)

    identity_matrix = B.eye(B.dtype(a), B.shape(a)[-1])
    # Deal with the batch dimensions.
    for i in range(B.rank(a) - 1):
        identity_matrix = B.expand_dims(identity_matrix, axis=0)
    # Use broadcasting to get the desired output.
    return B.expand_dims(a, axis=-1) * identity_matrix


@dispatch
def flatten(a: Numeric):  # pragma: no cover
    """Flatten a tensor.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Flattened tensor.
    """
    return reshape(a, -1)


def _vec_to_tril_side_upper_perm(m, offset=0):
    # Compute the length of a side of the square result.
    k = offset
    if k <= 0:
        side = int((math.sqrt(1 + 8 * m) - 1) / 2) - k
    else:
        side = int((math.sqrt(1 + 8 * (k * (k + 1) + m)) - (1 + 2 * k)) / 2)

    # Compute sorting permutation.
    ind_lower = np.tril_indices(side, k=offset)
    ind_upper = np.triu_indices(side, k=1 + offset)
    ind_concat = (
        np.concatenate((ind_lower[0], ind_upper[0])),
        np.concatenate((ind_lower[1], ind_upper[1])),
    )
    perm = np.lexsort((ind_concat[1], ind_concat[0]))

    return side, len(ind_upper[0]), perm


@dispatch
def vec_to_tril(a: Numeric, offset=0):
    """Construct a lower triangular matrix from a vector.

    Args:
        a (tensor): Vector.
        offset (int, optional): Diagonal offset.

    Returns:
        tensor: Lower triangular matrix.
    """
    if B.rank(a) < 1:
        raise ValueError("Input must be at least rank 1.")
    batch_shape = B.shape(a)[:-1]
    side, upper, perm = _vec_to_tril_side_upper_perm(B.shape(a)[-1], offset=offset)
    a = B.concat(a, B.zeros(B.dtype(a), *batch_shape, upper), axis=-1)
    return B.reshape(B.take(a, perm, axis=-1), *batch_shape, side, side)


@dispatch
def tril_to_vec(a: Numeric, offset=0):
    """Construct a vector from a lower triangular matrix.

    Args:
        a (tensor): Lower triangular matrix.
        offset (int, optional): Diagonal offset.

    Returns:
        tensor: Vector
    """
    if B.rank(a) < 2:
        raise ValueError("Input must be at least rank 2.")
    batch_shape = B.shape(a)[:-2]
    n, m = B.shape(a)[-2:]
    if n != m:
        raise ValueError("Input must be square.")
    indices = np.tril_indices(n, k=offset)
    # Convert to linear indices to be able to use `B.take`.
    indices = n * indices[0] + indices[1]
    return B.take(B.reshape(a, *batch_shape, n * n), indices, axis=-1)


@dispatch
@abstract()
def stack(*elements, **kw_args):  # pragma: no cover
    """Concatenate tensors along a new axis.

    Args:
        *elements (tensor): Tensors to stack.
        axis (int, optional): Index of new axis. Defaults to `0`.

    Returns:
        tensor: Stacked tensors.
    """


@dispatch
@abstract()
def unstack(a: Numeric, axis=0):  # pragma: no cover
    """Unstack tensors along an axis.

    Args:
        a (list): List of tensors.
        axis (int, optional): Index along which to unstack. Defaults to `0`.

    Returns:
        list[tensor]: List of tensors.
    """


@dispatch
@abstract()
def reshape(a: Numeric, *shape: Int):  # pragma: no cover
    """Reshape a tensor.

    Args:
        a (tensor): Tensor to reshape.
        *shape (shape): New shape.

    Returns:
        tensor: Reshaped tensor.
    """


@dispatch
@abstract()
def concat(*elements, **kw_args):  # pragma: no cover
    """Concatenate tensors along an axis.

    Args:
        *elements (tensor): Tensors to concatenate
        axis (int, optional): Axis along which to concatenate. Defaults to `0`.

    Returns:
        tensor: Concatenation.
    """


@dispatch
def concat2d(*rows: Union[list]):
    """Concatenate tensors into a matrix.

    Args:
        *rows (list[list[tensor]]): List of list of tensors, which form the
            rows of the matrix.

    Returns:
        tensor: Assembled matrix.
    """
    return concat(*[concat(*row, axis=1) for row in rows], axis=0)


@dispatch
@abstract()
def tile(a: Numeric, *repeats: Int):  # pragma: no cover
    """Tile a tensor.

    Args:
        a (tensor): Tensor to tile.
        *repeats (shape): Repetitions per dimension

    Returns:
        tensor: Tiled tensor.
    """


@dispatch
def take(a: Numeric, indices_or_mask, axis=0):  # pragma: no cover
    """Take particular elements along an axis.

    Args:
        a (tensor): Tensor.
        indices_or_mask (list): List of indices or boolean indicating which
            elements to take. Must be rank 1.
        axis (int, optional): Axis along which to take indices. Defaults to `0`.

    Returns:
        tensor: Selected subtensor.
    """
    if B.rank(indices_or_mask) != 1:
        raise ValueError("Indices or mask must be rank 1.")
    # Resolve negative axis specification.
    if axis < 0:
        axis += B.rank(a)
    # Construct slices.
    slices = tuple(
        indices_or_mask if i == axis else slice(None, None, None)
        for i in range(B.rank(a))
    )
    return a[slices]
