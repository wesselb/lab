# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from . import dispatch, B
from .types import Numeric, DType, default_dtype, Dimension
from .util import abstract

__all__ = ['nan',
           'pi',
           'log_2_pi',
           'isnan',
           'zeros',
           'ones',
           'eye',
           'linspace',
           'cast',
           'identity',
           'abs',
           'sign',
           'sqrt',
           'exp',
           'log',
           'sin',
           'cos',
           'tan',
           'tanh',
           'sigmoid',
           'relu',
           'add',
           'subtract',
           'multiply',
           'divide',
           'power',
           'minimum',
           'maximum',
           'leaky_relu',
           'min',
           'max',
           'sum',
           'mean',
           'std',
           'logsumexp',
           'all',
           'any',
           'lt',
           'le',
           'gt',
           'ge']

nan = np.nan  #: NaN.
pi = np.pi  #: Value of pi.
log_2_pi = np.log(2 * pi)  #: Value of log(2 * pi).


@dispatch(Numeric)
@abstract()
def isnan(a):  # pragma: no cover
    """Check whether a tensor is NaN.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor[bool]: `a` is NaN.
    """


@dispatch(DType, [Dimension])
@abstract(promote=None)
def zeros(dtype, *shape):  # pragma: no cover
    """Create a tensor of zeros.

    Can also give a reference tensor whose data type and shape will be used to
    construct a tensor of zeros.

    Args:
        dtype (dtype, optional): Data type. Defaults to
            `.types.default_dtype`.
        *shape (shape): Shape of the tensor.

    Returns:
        tensor: Matrix of zeros of shape `shape` and data type `dtype`.
    """


@dispatch.multi((int,),  # Single integer is not a reference.
                ([Dimension],))
def zeros(*shape):
    return zeros(default_dtype, *shape)


@dispatch(Numeric)
def zeros(ref):
    return zeros(B.dtype(ref), *B.shape(ref))


@dispatch(DType, [Dimension])
@abstract(promote=None)
def ones(dtype, *shape):  # pragma: no cover
    """Create a tensor of ones.

    Can also give a reference tensor whose data type and shape will be used to
    construct a tensor of ones.

    Args:
        dtype (dtype, optional): Data type. Defaults to
            `.types.default_dtype`.
        *shape (shape or ): Shape of the tensor.

    Returns:
        tensor: Matrix of ones of shape `shape` and data type `dtype`.
    """


@dispatch.multi((int,),  # Single integer is not a reference.
                ([Dimension],))
def ones(*shape):
    return ones(default_dtype, *shape)


@dispatch(Numeric)
def ones(ref):
    return ones(B.dtype(ref), *B.shape(ref))


@dispatch(DType, Dimension, Dimension)
@abstract(promote=None)
def eye(dtype, *shape):  # pragma: no cover
    """Create an identity matrix.

    Can also give a reference tensor whose data type and shape will be used to
    construct an identity matrix.

    Args:
        dtype (dtype, optional): Data type. Defaults to
            `.types.default_dtype`.
        *shape (shape): Shape of the matrix.

    Returns:
        tensor: Identity matrix of shape `shape` and data type `dtype`.
    """


@dispatch(DType, Dimension)
def eye(dtype, *shape):
    return eye(dtype, shape[0], shape[0])


@dispatch(Dimension, [Dimension])
def eye(*shape):
    return eye(default_dtype, *shape)


@dispatch.multi((int,),  # Single integer is not a reference.
                (Dimension,))
def eye(shape):
    return eye(default_dtype, shape, shape)


@dispatch(Numeric)
def eye(ref):
    return eye(B.dtype(ref), *B.shape(ref))


@dispatch(object, object, int)
@abstract(promote=2)
def linspace(a, b, num):
    """Create a vector of `c` numbers ranging from `a` to `c`, distributed
    linearly.

    Args:
        a (number): Lower bound.
        b (number): Upper bound.
        num (int): Number of numbers.

    Returns:
        vector: `c` numbers ranging from `a` to `c`, distributed linearly.
    """


@dispatch(Numeric, DType)
@abstract()
def cast(a, dtype):  # pragma: no cover
    """Cast an object to another data type.

    Args:
        a (tensor): Tensor to cast.
        dtype (dtype or tensor): New data type.

    Returns:
        tensor: `a`, but of data type `dtype`.
    """


@dispatch(Numeric, Numeric)
def cast(a, ref):
    return cast(a, B.dtype(ref))


# Unary functions:

@dispatch(Numeric)
@abstract()
def identity(a):  # pragma: no cover
    """Identity function

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: `a` exactly.
    """


@dispatch(Numeric)
@abstract()
def abs(a):  # pragma: no cover
    """Absolute value.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Absolute value of `a`.
    """


@dispatch(Numeric)
@abstract()
def sign(a):  # pragma: no cover
    """Sign function.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Sign of `a`.
    """


@dispatch(Numeric)
@abstract()
def sqrt(a):  # pragma: no cover
    """Square root.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Square root of `a`.
    """


@dispatch(Numeric)
@abstract()
def exp(a):  # pragma: no cover
    """Exponential function.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Exponential function evaluated at `a`.
    """


@dispatch(Numeric)
@abstract()
def log(a):  # pragma: no cover
    """Logarithmic function

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Logarithmic function evaluated at `a`.
    """


@dispatch(Numeric)
@abstract()
def sin(a):  # pragma: no cover
    """Sine function.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Sine function evaluated at `a`.
    """


@dispatch(Numeric)
@abstract()
def cos(a):  # pragma: no cover
    """Cosine function.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Cosine function evaluated at `a`.
    """


@dispatch(Numeric)
@abstract()
def tan(a):  # pragma: no cover
    """Tangent function.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Tangent function evaluated at `a`.
    """


@dispatch(Numeric)
@abstract()
def tanh(a):  # pragma: no cover
    """Tangent hyperbolic function.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Tangent hyperbolic function evaluated at `a`.
    """


@dispatch(object)
def sigmoid(a):
    """Sigmoid function.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Sigmoid function evaluated at `a`.
    """
    return 1 / (1 + exp(-a))


@dispatch(object)
def relu(a):
    """Rectified linear unit.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Rectified linear unit evaluated at `a`.
    """
    return maximum(B.zeros(a), a)


# Binary functions:


@dispatch(object, object)
@abstract(promote=2)
def add(a, b):  # pragma: no cover
    """Add two tensors.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor: Sum of `a` and `b`.
    """


@dispatch(object, object)
@abstract(promote=2)
def subtract(a, b):  # pragma: no cover
    """Subtract two tensors.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor: `a` minus `b`.
    """


@dispatch(object, object)
@abstract(promote=2)
def multiply(a, b):  # pragma: no cover
    """Multiply two tensors.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor: Product of `a` and `b`.
    """


@dispatch(object, object)
@abstract(promote=2)
def divide(a, b):  # pragma: no cover
    """Divide two tensors.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor: `a` divided by `b`.
    """


@dispatch(object, object)
@abstract(promote=2)
def power(a, power):  # pragma: no cover
    """Raise a tensor to a power.

    Args:
        a (tensor): Tensor.
        power (tensor): Power.

    Returns:
        tensor: `a` to the power of `power`.
    """


@dispatch(object, object)
@abstract(promote=2)
def minimum(a, b):  # pragma: no cover
    """Take the minimum of two tensors.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor: Minimum of `a` and `b`.
    """


@dispatch(object, object)
@abstract(promote=2)
def maximum(a, b):  # pragma: no cover
    """Take the maximum of two tensors.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor: Maximum of `a` and `b`.
    """


@dispatch(object, object)
def leaky_relu(a, alpha):  # pragma: no cover
    """Leaky rectified linear unit.

    Args:
        a (tensor): Input.
        alpha (tensor): Coefficient of leak.

    Returns:
        tensor: Activation value.
    """
    return maximum(multiply(a, alpha), a)


# Reductions:


@dispatch(Numeric)
@abstract()
def min(a, axis=None):  # pragma: no cover
    """Take the minimum of a tensor, possibly along an axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Optional axis.

    Returns:
        tensor: Reduced tensor.
    """


@dispatch(Numeric)
@abstract()
def max(a, axis=None):  # pragma: no cover
    """Take the maximum of a tensor, possibly along an axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Optional axis.

    Returns:
        tensor: Reduced tensor.
    """


@dispatch(Numeric)
@abstract()
def sum(a, axis=None):  # pragma: no cover
    """Sum a tensor, possibly along an axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Optional axis.

    Returns:
        tensor: Reduced tensor.
    """


@dispatch(Numeric)
@abstract()
def mean(a, axis=None):  # pragma: no cover
    """Take the mean of a tensor, possibly along an axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Optional axis.

    Returns:
        tensor: Reduced tensor.
    """


@dispatch(Numeric)
@abstract()
def std(a, axis=None):  # pragma: no cover
    """Compute the standard deviation of a tensor, possibly along an axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Optional axis.

    Returns:
        tensor: Reduced tensor.
    """


@dispatch(object)
def logsumexp(a, axis=None):  # pragma: no cover
    """Exponentiate a tensor, sum it, and then take the logarithm, possibly
    along an axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Optional axis.

    Returns:
        tensor: Reduced tensor.
    """
    a_max = max(a, axis=axis)
    # Put the axis back if one is specified.
    if axis is None:
        a_expanded = a_max
    else:
        a_expanded = B.expand_dims(a_max, axis=axis)
    return log(sum(exp(a - a_expanded), axis=axis)) + a_max


# Logical reductions:


@dispatch(Numeric)
@abstract()
def all(a, axis=None):  # pragma: no cover
    """Logical all of a tensor, possibly along an axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Optional axis.

    Returns:
        tensor: Reduced tensor.
    """


@dispatch(Numeric)
@abstract()
def any(a, axis=None):  # pragma: no cover
    """Logical any of a tensor, possibly along an axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Optional axis.

    Returns:
        tensor: Reduced tensor.
    """


# Logical comparisons:


@dispatch(object, object)
@abstract(promote=2)
def lt(a, b):  # pragma: no cover
    """Check whether one tensor is strictly less than another.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor[bool]: `a` is strictly less than `b`.
    """


@dispatch(object, object)
@abstract(promote=2)
def le(a, b):  # pragma: no cover
    """Check whether one tensor is less than or equal to another.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor[bool]: `a` is less than or equal to `b`.
    """


@dispatch(object, object)
@abstract(promote=2)
def gt(a, b):  # pragma: no cover
    """Check whether one tensor is strictly greater than another.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor[bool]: `a` is strictly greater than `b`.
    """


@dispatch(object, object)
@abstract(promote=2)
def ge(a, b):  # pragma: no cover
    """Check whether one tensor is greater than or equal to another.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor[bool]: `a` is greater than or equal to `b`.
    """
