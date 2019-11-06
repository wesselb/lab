import numpy as np
import tensorflow as tf

from . import dispatch, B, Numeric
from ..shaping import _vec_to_tril_shape_upper_perm
from ..types import Int

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
        raise ValueError('Argument must have rank 1 or 2.')


@dispatch(Numeric)
def vec_to_tril(a):
    if B.rank(a) != 1:
        raise ValueError('Input must be rank 1.')
    m, upper, perm = _vec_to_tril_shape_upper_perm(a)
    a = tf.concat((a, tf.zeros(upper, dtype=a.dtype)), axis=0)
    return tf.reshape(tf.gather(a, perm), [m, m])


@dispatch(Numeric)
def tril_to_vec(a):
    if B.rank(a) != 2:
        raise ValueError('Input must be rank 2.')
    n, m = a.shape
    if n != m:
        raise ValueError('Input must be square.')
    return tf.gather_nd(a, list(zip(*np.tril_indices(int(n)))))


@dispatch([Numeric])
def stack(*elements, **kw_args):
    return tf.stack(elements, axis=kw_args.get('axis', 0))


@dispatch(Numeric)
def unstack(a, axis=0):
    return tf.unstack(a, axis=axis)


@dispatch(Numeric, [Int])
def reshape(a, *shape):
    return tf.reshape(a, shape=shape)


@dispatch([Numeric])
def concat(*elements, **kw_args):
    return tf.concat(elements, axis=kw_args.get('axis', 0))


@dispatch(Numeric, [Int])
def tile(a, *repeats):
    return tf.tile(a, repeats)


@dispatch(Numeric, object)
def take(a, indices_or_mask, axis=0):
    if B.rank(indices_or_mask) != 1:
        raise ValueError('Indices or mask must be rank 1.')

    # Put axis `axis` first.
    if axis > 0:
        # Create a permutation to switch `axis` and `0`.
        perm = list(range(B.rank(a)))
        perm[0], perm[axis] = perm[axis], perm[0]
        a = tf.transpose(a, perm)

    # Figure out whether we're given indices or a mask.
    if isinstance(indices_or_mask, B.TF):
        mask = indices_or_mask.dtype == bool
    else:
        mask = len(indices_or_mask) > 0 and B.dtype(indices_or_mask[0]) == bool

    # Take the relevant part.
    if mask:
        a = tf.boolean_mask(a, indices_or_mask)
    else:
        a = tf.gather(a, indices_or_mask)

    # Put axis `axis` back again.
    if axis > 0:
        a = tf.transpose(a, perm)

    return a
