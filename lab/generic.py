# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np
from plum import Dispatcher, PromisedType, as_type

from . import B

epsilon = 1e-12  #: Magnitude of diagonal to regularise matrices with.

_Numeric = {Number, np.ndarray}  #: Type of numerical objects.
_DType = {type, np.dtype}  #: Type of data types.

_dispatch = Dispatcher()


def rank(a):
    """Get the rank of a tensor.

    Args:
        a (tensor): Tensor to get the rank of.
    """
    return B.shape_int(B.shape(a))[0]


def reg(a, diag=None, clip=True):
    """Add a diagonal to a matrix.

    Args:
        a (matrix): Matrix to add a diagonal to.
        diag (float, optional): Magnitude of the diagonal to add. Defaults to
            `default_reg_diag`.
        clip (bool, optional): Let `diag` be at least `default_reg_diag`.
            Defaults to `True`.
    """
    if diag is None:
        diag = epsilon
    elif clip:
        diag = B.maximum(diag, epsilon)
    return a + diag * B.eye(B.shape(a)[0], B.shape(a)[1], dtype=B.dtype(a))


@property
def log_2_pi():
    """Value of `log(2 * np.pi)`."""
    return B.log(B.array(2 * np.pi))


@property
def pi():
    """Value of `pi`."""
    return np.pi


@_dispatch(object, object)
def pw_dists2(a, b):
    """Compute the square of the pairwise Euclidean distances between design
    matrices.

    Args:
        a (design matrix, optional): First design matrix.
        b (design matrix, optional): Second design matrix. Defaults to first
            design matrix.
    """
    norms_a = B.sum(a ** 2, axis=1)[:, None]
    norms_b = B.sum(b ** 2, axis=1)[None, :]
    return norms_a + norms_b - 2 * B.matmul(a, b, tr_b=True)


@_dispatch(object)
def pw_dists2(a):
    norms = B.sum(a ** 2, axis=1)
    return norms[:, None] + norms[None, :] - 2 * B.matmul(a, a, tr_b=True)


@_dispatch(object, object)
def ew_dists2(a, b):
    return B.sum((a - b) ** 2, axis=1)[:, None]


@_dispatch([object])
def pw_dists(*args):
    """Compute the pairwise Euclidean distances between design matrices.

    Args:
        a (design matrix, optional): First design matrix.
        b (design matrix, optional): Second design matrix. Defaults to first
            design matrix.
    """
    d2 = pw_dists2(*args)
    # Clip at a bit higher than the smallest single-precision floating point
    # number.
    return B.sqrt(B.maximum(d2, B.cast(1e-30, dtype=B.dtype(d2))))


@_dispatch([object])
def ew_dists(*args):
    d2 = ew_dists2(*args)
    return B.sqrt(B.maximum(d2, B.cast(1e-30, dtype=B.dtype(d2))))


@_dispatch(Number)
def pw_dists(a):
    return B.array([[0.]])


@_dispatch(Number, Number)
def pw_dists(a, b):
    return B.array([[B.abs(a - b)]])


def is_scalar(a):
    """Check whether an object is a scalar.

    Args:
        a (tensor): Object to check.
    """
    return B.rank(a) == 0


@_dispatch(object, object)
def outer(a, b):
    """Outer product between two matrices.

    Args:
        a (matrix): First matrix.
        b (matrix, optional): Second matrix.
    """
    return B.matmul(a, b, tr_b=True)


@_dispatch(object)
def outer(a):
    return B.outer(a, a)


# Numeric type for dispatch has to be evaluated lazily.
class PromisedNumeric(PromisedType):
    def resolve(self):
        return B._Numeric


Numeric = PromisedNumeric()


# Type of data types for dispatch has to be evaluated lazily.
class PromisedDType(PromisedType):
    def resolve(self):
        return B._DType


DType = PromisedDType()


def dtype(a):
    """Get the data type of an object.

    Args:
        a (obj): Object to get data type of.
    """
    return B.array(a).dtype


def flatten(a):
    """Flatten an object.

    Args:
        a (tensor): Object to flatten.
    """
    return B.reshape(a, [-1])


def shape_int(a):
    """Get the shape of an object as a tuple of integers.

    Args:
        a (tensor): Object to get shape of.
    """
    return B.shape(a)


def length(a):
    """Get the number of elements in a tensor.

    Args:
        a (tensor): Tensor to get number of elements of.
    """
    num, shape = 1, B.shape(a)
    for i in range(B.rank(a)):
        num *= shape[i]
    return num


def issubdtype(a, b):
    """Check if `a` is a sub-data-type of `b`.

    Args:
        a (dtype): Data type.
        b (dtype): Data type to check membership of.
    """
    return np.issubdtype(a, b)


class _PseudoInstance(object):
    """A pseudo-instance to facilitate dispatch.

    Args:
        type (type): Type of pseudo-instance.
    """

    def __init__(self, type):
        self.type = type


@_dispatch(object)
def _get_type(x):
    return as_type(type(x))


@_dispatch(_PseudoInstance)
def _get_type(x):
    return as_type(x.type)


@_dispatch(object, object)
def promotion_rule(obj1, obj2):
    """Promotion rule.

    Args:
        obj1 (object): Object of first type to promote.
        obj2 (object): Object of second type to promote.

    Returns:
        type: Type to convert to.
    """
    if _get_type(obj1) == _get_type(obj2):
        return _get_type(obj1)
    else:
        raise RuntimeError('No promotion rule for "{}" and "{}".'
                           ''.format(_get_type(obj1), _get_type(obj2)))


@_dispatch(object, object, object)
def add_promotion_rule(type1, type2, type_to):
    """Add a promotion rule.

    Args:
        type1 (type): First type to promote.
        type2 (type): Second type to promote.
        type_to (type): Type to convert to.
    """
    promotion_rule.extend(type1, type2)(lambda t1, t2: type_to)
    if as_type(type1) != as_type(type2):
        promotion_rule.extend(type2, type1)(lambda t1, t2: type_to)


@_dispatch(object, object)
def convert(obj_to_convert, obj_from_target):
    """Convert an object to a particular type.

    Args:
        obj_to_convert (object): Object to convert.
        obj_from_target (object): Object from type to convert to.

    Returns:
        object: `object_to_covert` converted to type of `obj_from_target`.
    """
    if _get_type(obj_to_convert) <= _get_type(obj_from_target):
        return obj_to_convert
    else:
        raise RuntimeError('No rule to convert a "{}" to a "{}".'
                           ''.format(_get_type(obj_to_convert),
                                     _get_type(obj_from_target)))


@_dispatch(object, object, [object])
def promote(*objs):
    """Promote objects to a common type.

    Args:
        *objs (object): Objects to convert.

    Returns:
        tuple: `objs`, but all converted to a common type.
    """
    # Find the common type.
    common_type = promotion_rule(objs[0], objs[1])
    for obj in objs[2:]:
        method = promotion_rule.invoke(common_type, type(obj))
        common_type = method(_PseudoInstance(common_type), obj)

    # Convert objects.
    converted_objs = []
    for obj in objs:
        method = convert.invoke(type(obj), common_type)
        converted_objs.append(method(obj, _PseudoInstance(common_type)))

    return tuple(converted_objs)


@_dispatch(object)
def promote(*objs):
    return objs
