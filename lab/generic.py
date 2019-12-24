import numpy as np
from plum import Callable, convert, add_conversion_method

from . import dispatch, B, Dispatcher
from .types import (
    Number,
    Numeric,
    DType,
    Int,
    NPNumeric,
    AGNumeric,
    TFNumeric,
    TorchNumeric,
)
from .util import abstract

__all__ = ['nan',
           'pi',
           'log_2_pi',
           'isnan',
           'zeros',
           'ones',
           'zero',
           'one',
           'eye',
           'linspace',
           'range',
           'cast',
           'identity',
           'negative',
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
           'softplus',
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
           'ge',
           'bvn_cdf',
           'scan',
           'sort',
           'argsort',
           'to_numpy']

_dispatch = Dispatcher()

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


@dispatch(DType, [Int])
@abstract(promote=None)
def zeros(dtype, *shape):  # pragma: no cover
    """Create a tensor of zeros.

    Can also give a reference tensor whose data type and shape will be used to
    construct a tensor of zeros.

    Args:
        dtype (dtype, optional): Data type. Defaults to the default data type.
        *shape (shape): Shape of the tensor.

    Returns:
        tensor: Tensor of zeros of shape `shape` and data type `dtype`.
    """


@dispatch.multi((Int,),  # Single integer is not a reference.
                ([Int],))
def zeros(*shape):
    return zeros(B.default_dtype, *shape)


@dispatch(Numeric)
def zeros(ref):
    return zeros(B.dtype(ref), *B.shape(ref))


@dispatch(DType, [Int])
@abstract(promote=None)
def ones(dtype, *shape):  # pragma: no cover
    """Create a tensor of ones.

    Can also give a reference tensor whose data type and shape will be used to
    construct a tensor of ones.

    Args:
        dtype (dtype, optional): Data type. Defaults to the default data type.
        *shape (shape): Shape of the tensor.

    Returns:
        tensor: Tensor of ones of shape `shape` and data type `dtype`.
    """


@dispatch.multi((Int,),  # Single integer is not a reference.
                ([Int],))
def ones(*shape):
    return ones(B.default_dtype, *shape)


@dispatch(Numeric)
def ones(ref):
    return ones(B.dtype(ref), *B.shape(ref))


@dispatch(DType)
def zero(dtype):
    """Create a `0` with a particular data type.

    Args:
        dtype (dtype): Data type.

    Returns:
        scalar: `0` with data type `dtype`.
    """
    return B.cast(dtype, 0)


@dispatch(Numeric)
def zero(ref):
    return zero(B.dtype(ref))


@dispatch(DType)
def one(dtype):
    """Create a `1` with a particular data type.

    Args:
        dtype (dtype): Data type.

    Returns:
        scalar: `1` with data type `dtype`.
    """
    return B.cast(dtype, 1)


@dispatch(Numeric)
def one(ref):
    return one(B.dtype(ref))


@dispatch(DType, Int, Int)
@abstract(promote=None)
def eye(dtype, *shape):  # pragma: no cover
    """Create an identity matrix.

    Can also give a reference tensor whose data type and shape will be used to
    construct an identity matrix.

    Args:
        dtype (dtype, optional): Data type. Defaults to the default data type.
        *shape (shape): Shape of the matrix.

    Returns:
        tensor: Identity matrix of shape `shape` and data type `dtype`.
    """


@dispatch(DType, Int)
def eye(dtype, *shape):
    return eye(dtype, shape[0], shape[0])


@dispatch(Int, [Int])
def eye(*shape):
    return eye(B.default_dtype, *shape)


@dispatch(Int)  # Single integer is not a reference.
def eye(shape):
    return eye(B.default_dtype, shape, shape)


@dispatch(Numeric)
def eye(ref):
    return eye(B.dtype(ref), *B.shape(ref))


@dispatch(DType, object, object, Int)
@abstract(promote=None)
def linspace(dtype, a, b, num):
    """Create a vector of `c` numbers ranging from `a` to `c`, distributed
    linearly.

    Args:
        dtype (dtype, optional): Data type. Defaults to the default data type.
        a (number): Lower bound.
        b (number): Upper bound.
        num (int): Number of numbers.

    Returns:
        vector: `c` numbers ranging from `a` to `c`, distributed linearly.
    """


@dispatch(object, object, Int)
def linspace(a, b, num):
    return linspace(B.default_dtype, a, b, num)


@dispatch(DType, object, object, object)
@abstract(promote=None)
def range(dtype, start, stop, step):
    """Create a vector of numbers ranging from `start` to `stop` with step
    size `step`.

    Args:
        dtype (dtype, optional): Data type. Defaults to `int`.
        start (number, optional): Start of range. Defaults to `0`.
        stop (number): End of range.
        step (number, optional): Step size. Defaults to `1`.

    Returns:
        vector: Numbers ranging from `start` to `stop` with step size `step`.
    """


@dispatch(object, object, object)
def range(start, stop, step):
    return range(int, start, stop, step)


@dispatch(DType, object, object)
def range(dtype, start, stop):
    return range(dtype, start, stop, 1)


@dispatch(object, object)
def range(start, stop):
    return range(int, start, stop, 1)


@dispatch(DType, object)
def range(dtype, stop):
    return range(dtype, 0, stop, 1)


@dispatch(object)
def range(stop):
    return range(int, 0, stop, 1)


@dispatch(Numeric, DType)
@abstract(promote=None)
def cast(dtype, a):  # pragma: no cover
    """Cast an object to another data type.

    Args:
        dtype (dtype): New data type.
        a (tensor): Tensor to cast.

    Returns:
        tensor: `a`, but of data type `dtype`.
    """


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
def negative(a):  # pragma: no cover
    """Negate a tensor.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Negative of `a`.
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
def softplus(a):
    """Softplus function.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Softplus function evaluated at `a`.
    """
    zero = B.cast(B.dtype(a), 0)
    return log(1 + exp(-abs(a))) + maximum(a, zero)


@dispatch(object)
def relu(a):
    """Rectified linear unit.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Rectified linear unit evaluated at `a`.
    """
    zero = B.cast(B.dtype(a), 0)
    return maximum(zero, a)


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


@dispatch(object, object, object)
@abstract(promote=3)
def bvn_cdf(a, b, c):
    """Standard bivariate normal CDF. Computes the probability that `X < a`
    and `Y < b` if `X ~ N(0, 1)`, `Y ~ N(0, 1)`, and `X` and `Y` have
    correlation `c`.

    Args:
        a (tensor): First upper limit. Must be a rank-one tensor.
        b (tensor): Second upper limit. Must be a rank-one tensor.
        c (tensor): Correlation coefficient. Must be a rank-one tensor.

    Returns:
        tensor: Probabilities of the same shape as the input.
    """


@_dispatch(object)
def _as_tuple(x):
    return x,


@_dispatch(tuple)
def _as_tuple(x):
    return x


@_dispatch(tuple)
def _compress(x):
    if len(x) == 1:
        return x[0]
    else:
        return x


@dispatch(Callable, object, [object])
def scan(f, xs, *init_state):
    """Perform a TensorFlow-style scanning operation.

    Args:
        f (function): Scanning function.
        xs (tensor): Tensor to scan over.
        *init_state (tensor): Initial state.
    """
    state = init_state
    state_shape = [B.shape(s) for s in state]
    states = []

    # Cannot simply iterate, because that breaks TensorFlow.
    for i in range(int(B.shape(xs)[0])):

        state = _as_tuple(f(_compress(state), xs[i]))
        new_state_shape = [B.shape(s) for s in state]

        # Check that the state shape remained constant.
        if new_state_shape != state_shape:
            raise RuntimeError('Shape of state changed from {} to {}.'
                               ''.format(state_shape, new_state_shape))

        # Record the state, stacked over the various elements.
        states.append(B.stack(*state, axis=0))

    # Stack states over iterations.
    states = B.stack(*states, axis=0)

    # Put the elements dimension first and return.
    return B.transpose(states, perm=(1, 0) + tuple(range(2, B.rank(states))))


@dispatch(Numeric)
@abstract(promote=None)
def sort(a, axis=-1, descending=False):
    """Sort a tensor along an axis in ascending order.

    Args:
        a (tensor): Tensor to sort.
        axis (int, optional): Axis to sort along. Defaults to `-1`.
        descending (bool, optional): Sort in descending order. Defaults to
            `False`.

    Returns:
        tensor: `a`, but sorted.
    """


@dispatch(Numeric)
@abstract(promote=None)
def argsort(a, axis=-1, descending=False):
    """Get the indices that would a tensor along an axis in ascending order.

    Args:
        a (tensor): Tensor to sort.
        axis (int, optional): Axis to sort along. Defaults to `-1`.
        descending (bool, optional): Sort in descending order. Defaults to
            `False`.

    Returns:
        tensor: The indices that would sort `a`.
    """


NPOrNum = {NPNumeric, Number}  #: Type NumPy numeric or number.
add_conversion_method(AGNumeric, NPOrNum, lambda x: x._value)
add_conversion_method(TFNumeric, NPOrNum, lambda x: x.numpy())
add_conversion_method(TorchNumeric, NPOrNum, lambda x: x.detach().numpy())


@dispatch(object)
def to_numpy(a):
    """Convert an object to NumPy.

    Args:
        a (object): Object to convert.

    Returns:
        `np.ndarray`: `a` as NumPy.
    """
    return convert(a, NPOrNum)


@dispatch([object])
def to_numpy(*elements):
    return to_numpy(elements)


@dispatch(list)
def to_numpy(a):
    return [to_numpy(x) for x in a]


@dispatch(tuple)
def to_numpy(a):
    return tuple(to_numpy(x) for x in a)


@dispatch(dict)
def to_numpy(a):
    return {k: to_numpy(v) for k, v in a.items()}
