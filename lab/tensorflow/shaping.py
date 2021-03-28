import tensorflow as tf

from . import dispatch, B, Numeric
from ..types import Int, TFNumeric, NPNumeric

__all__ = []


@dispatch(Numeric)
def length(a):
    return tf.size(a)


@dispatch(Numeric)
def expand_dims(a, axis=0):
    return tf.expand_dims(a, axis=axis)


@dispatch(Numeric)
def squeeze(a):
    return tf.squeeze(a)


@dispatch(Numeric)
def diag(a):
    if B.rank(a) == 1:
        return tf.linalg.diag(a)
    elif B.rank(a) == 2:
        return tf.linalg.diag_part(a)
    else:
        raise ValueError("Input must have rank 1 or 2.")


@dispatch(Numeric)
def diag_extract(a):
    return tf.linalg.diag_part(a)


@dispatch(TFNumeric)  # This function already has a generic implementation.
def diag_construct(a):
    return tf.linalg.diag(a)


@dispatch([Numeric])
def stack(*elements, axis=0):
    return tf.stack(elements, axis=axis)


@dispatch(Numeric)
def unstack(a, axis=0):
    return tf.unstack(a, axis=axis)


@dispatch(Numeric, [Int])
def reshape(a, *shape):
    return tf.reshape(a, shape=shape)


@dispatch([Numeric])
def concat(*elements, axis=0):
    return tf.concat(elements, axis=axis)


@dispatch(Numeric, [Int])
def tile(a, *repeats):
    return tf.tile(a, repeats)


@dispatch(TFNumeric, object)  # This function already has a generic implementation.
def take(a, indices_or_mask, axis=0):
    if B.rank(indices_or_mask) != 1:
        raise ValueError("Indices or mask must be rank 1.")
    is_mask, indices_or_mask = _is_mask_and_convert(indices_or_mask)
    if is_mask:
        return tf.boolean_mask(a, indices_or_mask, axis=axis)
    else:
        return tf.gather(a, indices_or_mask, axis=axis)


@dispatch(TFNumeric)
def _is_mask_and_convert(indices_or_mask):
    return indices_or_mask.dtype == bool, indices_or_mask


@dispatch(NPNumeric)
def _is_mask_and_convert(indices_or_mask):
    return indices_or_mask.dtype == bool, tf.constant(indices_or_mask)


@dispatch({tuple, list})
def _is_mask_and_convert(indices_or_mask):
    return B.dtype(indices_or_mask[0]) == bool, indices_or_mask
