import tensorflow as tf
from plum import Union

from . import dispatch, B, Numeric
from ..types import Int, TFNumeric, NPNumeric

__all__ = []


@dispatch
def length(a: Numeric):
    return tf.size(a)


@dispatch
def expand_dims(a: Numeric, axis=0):
    return tf.expand_dims(a, axis=axis)


@dispatch
def squeeze(a: Numeric):
    return tf.squeeze(a)


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
def stack(*elements: Numeric, axis=0):
    return tf.stack(elements, axis=axis)


@dispatch
def unstack(a: Numeric, axis=0):
    return tf.unstack(a, axis=axis)


@dispatch
def reshape(a: Numeric, *shape: Int):
    return tf.reshape(a, shape=shape)


@dispatch
def concat(*elements: Numeric, axis=0):
    return tf.concat(elements, axis=axis)


@dispatch
def tile(a: Numeric, *repeats: Int):
    return tf.tile(a, repeats)


@dispatch
def take(a: TFNumeric, indices_or_mask, axis=0):
    if B.rank(indices_or_mask) != 1:
        raise ValueError("Indices or mask must be rank 1.")
    is_mask, indices_or_mask = _is_mask_and_convert(indices_or_mask)
    if is_mask:
        return tf.boolean_mask(a, indices_or_mask, axis=axis)
    else:
        return tf.gather(a, indices_or_mask, axis=axis)


@dispatch
def _is_mask_and_convert(indices_or_mask: TFNumeric):
    return indices_or_mask.dtype == bool, indices_or_mask


@dispatch
def _is_mask_and_convert(indices_or_mask: NPNumeric):
    return indices_or_mask.dtype == bool, tf.constant(indices_or_mask)


@dispatch
def _is_mask_and_convert(indices_or_mask: Union[tuple, list]):
    return B.dtype(indices_or_mask[0]) == bool, indices_or_mask
