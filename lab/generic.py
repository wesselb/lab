# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np
from plum import Dispatcher, PromisedType, kind, type_parameter

from . import B

epsilon = 1e-12  #: Magnitude of diagonal to regularise matrices with.

_Numeric = {Number, np.ndarray}  #: Type of numerical objects.

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


Type = kind()  #: A type to be used in dispatch.


@_dispatch(Type, Type)
def promotion_rule(type1, type2):
    """Promotion rule.

    Args:
        type1 (:class:`.generic.Type`): First type to promote.
        type2 (:class:`.generic.Type`): Second type to promote.

    Returns:
        type: Type to convert to.
    """
    if type_parameter(type(type1)) == type_parameter(type(type2)):
        return type_parameter(type(type1))
    else:
        raise RuntimeError('No promotion rule for "{}" and "{}".'
                           ''.format(type_parameter(type(type1)).__name__,
                                     type_parameter(type(type2)).__name__))


@_dispatch(type, type, type)
def add_promotion_rule(type1, type2, type_to):
    """Add a promotion rule.

    Args:
        type1 (:class:`.generic.Type`): First type to promote.
        type2 (:class:`.generic.Type`): Second type to promote.
        type_to (:class:`.generic.Type`): Type to convert to.
    """
    promotion_rule.extend(Type(type1), Type(type2))(lambda t1, t2: type_to)
    promotion_rule.extend(Type(type2), Type(type1))(lambda t1, t2: type_to)


@_dispatch(Type, object)
def convert(target_type, obj_to_convert):
    """Convert an object to a particular type.

    Args:
        target_type (:class:`.generic.Type`): Type to convert to.
        obj_to_convert (object): Object to convert.

    Returns:
        object: `object_to_covert` converted to `target_type`.
    """
    if type(obj_to_convert) == type_parameter(type(target_type)):
        return obj_to_convert
    else:
        target_type_name = type_parameter(type(target_type)).__name__
        raise RuntimeError('No conversion rule to convert a "{}" to a "{}".'
                           ''.format(type(obj_to_convert).__name__,
                                     target_type_name))


@_dispatch(object, object, [object])
def promote(*objs):
    """Promote objects to a common type.

    Args:
        *objs (object): Objects to convert.

    Returns:
        tuple: `objs`, but all converted to a common type.
    """
    common_type = promotion_rule(Type(type(objs[0]))(), Type(type(objs[1]))())
    for obj in objs[2:]:
        common_type = promotion_rule(Type(common_type)(), Type(type(obj))())
    return tuple(convert(Type(common_type)(), obj) for obj in objs)


@_dispatch(object)
def promote(*objs):
    return objs
