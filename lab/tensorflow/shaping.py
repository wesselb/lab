from plum import Union

import tensorflow as tf

from ..shape import unwrap_dimension
from ..types import Int, NPNumeric, TFNumeric
from ..util import resolve_axis
from . import B, Numeric, dispatch

__all__ = []


@dispatch
def length(a: Numeric):
    return tf.size(a)


@dispatch
def _expand_dims(a: Numeric, axis: Int = 0):
    return tf.expand_dims(a, axis=axis)


@dispatch
def squeeze(a: Numeric, axis: Union[Int, None] = None):
    return tf.squeeze(a, axis=axis)


@dispatch
def broadcast_to(a: Numeric, *shape: Int):
    return tf.broadcast_to(a, shape)


@dispatch
def diag(a: Numeric):
    if B.rank(a) == 1:
        return tf.linalg.diag(a)
    elif B.rank(a) == 2:
        return tf.linalg.diag_part(a)
    else:
        raise ValueError("Input must have rank 1 or 2.")


@dispatch
def diag_extract(a: Numeric):
    return tf.linalg.diag_part(a)


@dispatch
def diag_construct(a: TFNumeric):
    return tf.linalg.diag(a)


@dispatch
def stack(*elements: Numeric, axis: Int = 0):
    return tf.stack(elements, axis=axis)


@dispatch
def _unstack(a: Numeric, axis: Int = 0):
    return tf.unstack(a, axis=axis)


@dispatch
def reshape(a: Numeric, *shape: Int):
    return tf.reshape(a, shape=shape)


@dispatch
def concat(*elements: Numeric, axis: Int = 0):
    return tf.concat(elements, axis=axis)


@dispatch
def tile(a: Numeric, *repeats: Int):
    return tf.tile(a, repeats)


@dispatch
def take(a: TFNumeric, indices_or_mask, axis: Int = 0):
    if B.rank(indices_or_mask) != 1:
        raise ValueError("Indices or mask must be rank 1.")
    is_mask, indices_or_mask, shape_hint = _is_mask_convert_shape_hint(indices_or_mask)

    # Perform taking operation.
    if is_mask:
        # `tf.boolean_mask` isn't happy with negative axes.
        result = tf.boolean_mask(a, indices_or_mask, axis=resolve_axis(a, axis))
    else:
        result = tf.gather(a, indices_or_mask, axis=axis)

    # Apply the shape hint, if it is available.
    if shape_hint is not None:
        # Carefully unwrap to deal with lazy shapes.
        shape = list(map(unwrap_dimension, B.shape(a)))
        shape[axis] = shape_hint
        result.set_shape(shape)

    return result


@dispatch
def _is_mask_convert_shape_hint(indices_or_mask: TFNumeric):
    return indices_or_mask.dtype == bool, indices_or_mask, None


@dispatch
def _is_mask_convert_shape_hint(indices_or_mask: NPNumeric):
    is_mask = indices_or_mask.dtype == bool
    if is_mask:
        shape_hint = sum(indices_or_mask)
    else:
        shape_hint = len(indices_or_mask)
    return is_mask, tf.constant(indices_or_mask), shape_hint


@dispatch
def _is_mask_convert_shape_hint(indices_or_mask: Union[tuple, list]):
    if len(indices_or_mask) == 0:
        # Treat an empty tuple or list as a list of no indices. The data type does not
        # matter, except that it must be integer.
        return False, tf.constant([], dtype=tf.int32), 0
    else:
        is_mask = B.dtype(indices_or_mask[0]) == bool
        if is_mask:
            shape_hint = sum(indices_or_mask)
        else:
            shape_hint = len(indices_or_mask)
        return is_mask, indices_or_mask, shape_hint
