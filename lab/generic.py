from functools import wraps
import warnings
from types import FunctionType
from typing import Callable

import numpy as np
from plum import convert, add_conversion_method
from plum.type import VarArgs, Union

from . import dispatch, B, Dispatcher
from .control_flow import control_flow
from .types import (
    Number,
    Numeric,
    DType,
    Int,
    NPNumeric,
    AGNumeric,
    TFNumeric,
    TorchNumeric,
    JAXNumeric,
)
from .util import abstract

__all__ = [
    "nan",
    "pi",
    "log_2_pi",
    "isabstract",
    "jit",
    "isnan",
    "Device",
    "device",
    "on_device",
    "set_global_device",
    "to_active_device",
    "zeros",
    "ones",
    "zero",
    "one",
    "eye",
    "linspace",
    "range",
    "cast",
    "identity",
    "negative",
    "abs",
    "sign",
    "sqrt",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "tanh",
    "erf",
    "sigmoid",
    "softplus",
    "relu",
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "minimum",
    "maximum",
    "leaky_relu",
    "min",
    "argmin",
    "max",
    "argmax",
    "sum",
    "nansum",
    "mean",
    "nanmean",
    "std",
    "nanstd",
    "logsumexp",
    "all",
    "any",
    "lt",
    "le",
    "gt",
    "ge",
    "bvn_cdf",
    "cond",
    "where",
    "scan",
    "sort",
    "argsort",
    "quantile",
    "to_numpy",
]

_dispatch = Dispatcher()

nan = np.nan  #: NaN.
pi = np.pi  #: Value of pi.
log_2_pi = np.log(2 * pi)  #: Value of log(2 * pi).


@dispatch
@abstract()
def isabstract(a: Numeric):  # pragma: no cover
    """Check whether a tensor is abstract.

    Args:
        a (tensor): Tensor.

    Returns:
        bool: `True` if `a` is abstract, otherwise `False`.
    """


class JittedFunction:
    """A function that will be compiled just-in-time.

    Args:
        f_python (function): Python function to compile.
        jit_kw_args (dict): Keyword arguments to pass to the JIT.
    """

    def __init__(self, f_python, jit_kw_args):
        self._f_python = f_python
        self._compilation_cache = {}
        self._jit_kw_args = jit_kw_args

    def __call__(self, *args, **kw_args):
        return _jit_run(
            self._f_python,
            self._compilation_cache,
            self._jit_kw_args,
            *args,
            **kw_args,
        )


def jit(f: FunctionType = None, **kw_args):
    """Decorator to compile a function just-in-time.

    Further takes in keyword arguments which will be passed to the JIT.

    Args:
        f (function): Function to compile just-in-time.
    """
    # Support partial setting of `**kw_args`.
    if f is None:

        def dec(f_):
            return jit(f_, **kw_args)

        return dec

    # Use a control flow cache and lazy shapes to make sure that the JIT compilation
    # doesn't evaluate abstract tensors.

    cache = B.ControlFlowCache()

    @wraps(f)
    def f_safe(*args_, **kw_args_):
        with cache:
            with B.lazy_shapes():
                return f(*args_, **kw_args_)

    # The function needs to be run once before it can safe be compiled. That will be
    # handled by :func:`._jit_run`.

    return JittedFunction(f_safe, jit_kw_args=kw_args)


@dispatch
@abstract()
def _jit_run(
    f: FunctionType,
    compilation_cache: dict,
    jit_kw_args: dict,
    *args: Numeric,
    **kw_args
):  # pragma: no cover
    pass


@dispatch
@abstract()
def isnan(a: Numeric):  # pragma: no cover
    """Check whether a tensor is NaN.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor[bool]: `a` is NaN.
    """


class Device:
    """Context manager that tracks and changes the active device.

    Args:
        name (str): Name of the device.

    Attributes:
        active_name (str or :obj:`None`): Name of the active device.
        name (str): Name of the device.
    """

    active_name = None
    _tf_manager = None

    def __init__(self, name):
        self.name = name
        self._active_tf_manager = None

    def __enter__(self):
        # Set active name.
        Device.active_name = self.name

        # Activate the TF device manager, if it is available.
        if Device._tf_manager:
            self._active_tf_manager = Device._tf_manager(self.name)
            self._active_tf_manager.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Unset the active name.
        Device.active_name = None

        # Exit the TF device manager, if it was entered.
        if self._active_tf_manager:
            self._active_tf_manager.__exit__(exc_type, exc_val, exc_tb)


@dispatch
@abstract()
def device(a: Numeric):
    """Get the device on which a tensor lives.

    Args:
        a (tensor): Tensor to get device of.

    Returns:
        str: Device of `a`.
    """


@dispatch
def device(device: str):  # pragma: no cover
    warnings.warn(
        "The use of `device` to change the active device is deprecated. Please use "
        "`on_device` instead.",
        category=DeprecationWarning,
    )
    return on_device(device)


@dispatch
def on_device(device):
    """Create a context to change the active device.

    Args:
        a (device): New active device.

    Returns:
        :class:`.Device`: Context to change the active device.
    """
    return on_device(str(device))


@dispatch
def on_device(device: str):
    return Device(device)


@dispatch
def set_global_device(device):
    """Change the active device globally.

    Args:
        a (device): New active device.
    """
    on_device(device).__enter__()


@dispatch
@abstract()
def to_active_device(a: Numeric):  # pragma: no cover
    """Move a tensor to the active device.

    Args:
        a (tensor): Tensor to move.

    Returns:
        tensor: `a` on the active device.
    """


@dispatch
def to_active_device(a: Number):
    return a


@dispatch
@abstract()
def zeros(dtype: DType, *shape: Int):  # pragma: no cover
    """Create a tensor of zeros.

    Can also give a reference tensor whose data type and shape will be used to
    construct a tensor of zeros.

    Args:
        dtype (dtype, optional): Data type. Defaults to the default data type.
        *shape (shape): Shape of the tensor.

    Returns:
        tensor: Tensor of zeros of shape `shape` and data type `dtype`.
    """


@dispatch.multi((Int,), (VarArgs(Int),))  # Single integer is not a reference.
def zeros(*shape: Int):
    return zeros(B.default_dtype, *shape)


@dispatch
def zeros(ref: Numeric):
    return zeros(B.dtype(ref), *B.shape(ref))


@dispatch
@abstract()
def ones(dtype: DType, *shape: Int):  # pragma: no cover
    """Create a tensor of ones.

    Can also give a reference tensor whose data type and shape will be used to
    construct a tensor of ones.

    Args:
        dtype (dtype, optional): Data type. Defaults to the default data type.
        *shape (shape): Shape of the tensor.

    Returns:
        tensor: Tensor of ones of shape `shape` and data type `dtype`.
    """


@dispatch.multi((Int,), (VarArgs(Int),))  # Single integer is not a reference.
def ones(*shape: Int):
    return ones(B.default_dtype, *shape)


@dispatch
def ones(ref: Numeric):
    return ones(B.dtype(ref), *B.shape(ref))


@dispatch
def zero(dtype: DType):
    """Create a `0` with a particular data type.

    Args:
        dtype (dtype): Data type.

    Returns:
        scalar: `0` with data type `dtype`.
    """
    return B.cast(dtype, 0)


@dispatch
def zero(ref: Numeric):
    return zero(B.dtype(ref))


@dispatch
def one(dtype: DType):
    """Create a `1` with a particular data type.

    Args:
        dtype (dtype): Data type.

    Returns:
        scalar: `1` with data type `dtype`.
    """
    return B.cast(dtype, 1)


@dispatch
def one(ref: Numeric):
    return one(B.dtype(ref))


@dispatch
def eye(dtype: DType, *shape: Int):  # pragma: no cover
    """Create an identity matrix.

    Can also give a reference tensor whose data type and shape will be used to
    construct an identity matrix.

    Args:
        dtype (dtype, optional): Data type. Defaults to the default data type.
        *shape (shape): Shape of the matrix.

    Returns:
        tensor: Identity matrix of shape `shape` and data type `dtype`.
    """
    if len(shape) == 2:
        return _eye2(dtype, *shape)
    else:
        # It must be that `len(shape) > 2`.
        identity_matrix = _eye2(dtype, *shape[-2:])
        batch_shape = shape[:-2]
        for _ in range(len(batch_shape)):
            identity_matrix = B.expand_dims(identity_matrix, axis=0)
        return B.tile(identity_matrix, *batch_shape, 1, 1)


@dispatch
@abstract()
def _eye2(dtype: DType, *shape: Int):  # pragma: no cover
    pass


@dispatch
def eye(dtype: DType, shape: Int):
    return eye(dtype, shape, shape)


@dispatch
def eye(*shape: Int):
    return eye(B.default_dtype, *shape)


@dispatch
def eye(shape: Int):
    return eye(B.default_dtype, shape, shape)


@dispatch
def eye(ref: Numeric):
    return eye(B.dtype(ref), *B.shape(ref))


@dispatch
@abstract()
def linspace(dtype: DType, a, b, num: Int):
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


@dispatch
def linspace(a, b, num: Int):
    return linspace(B.default_dtype, a, b, num)


@dispatch
@abstract()
def range(dtype: DType, start, stop, step):
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


@dispatch
def range(start, stop, step):
    return range(int, start, stop, step)


@dispatch
def range(dtype: DType, start, stop):
    return range(dtype, start, stop, 1)


@dispatch
def range(start, stop):
    return range(int, start, stop, 1)


@dispatch
def range(dtype: DType, stop):
    return range(dtype, 0, stop, 1)


@dispatch
def range(stop):
    return range(int, 0, stop, 1)


@dispatch
@abstract()
def cast(dtype: Numeric, a: DType):  # pragma: no cover
    """Cast an object to another data type.

    Args:
        dtype (dtype): New data type.
        a (tensor): Tensor to cast.

    Returns:
        tensor: `a`, but of data type `dtype`.
    """


# Unary functions:


@dispatch
@abstract()
def identity(a: Numeric):  # pragma: no cover
    """Identity function

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: `a` exactly.
    """


@dispatch
@abstract()
def negative(a: Numeric):  # pragma: no cover
    """Negate a tensor.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Negative of `a`.
    """


@dispatch
@abstract()
def abs(a: Numeric):  # pragma: no cover
    """Absolute value.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Absolute value of `a`.
    """


@dispatch
@abstract()
def sign(a: Numeric):  # pragma: no cover
    """Sign function.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Sign of `a`.
    """


@dispatch
@abstract()
def sqrt(a: Numeric):  # pragma: no cover
    """Square root.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Square root of `a`.
    """


@dispatch
@abstract()
def exp(a: Numeric):  # pragma: no cover
    """Exponential function.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Exponential function evaluated at `a`.
    """


@dispatch
@abstract()
def log(a: Numeric):  # pragma: no cover
    """Logarithmic function

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Logarithmic function evaluated at `a`.
    """


@dispatch
@abstract()
def sin(a: Numeric):  # pragma: no cover
    """Sine function.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Sine function evaluated at `a`.
    """


@dispatch
@abstract()
def cos(a: Numeric):  # pragma: no cover
    """Cosine function.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Cosine function evaluated at `a`.
    """


@dispatch
@abstract()
def tan(a: Numeric):  # pragma: no cover
    """Tangent function.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Tangent function evaluated at `a`.
    """


@dispatch
@abstract()
def tanh(a: Numeric):  # pragma: no cover
    """Tangent hyperbolic function.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Tangent hyperbolic function evaluated at `a`.
    """


@dispatch
@abstract()
def erf(a: Numeric):  # pragma: no cover
    """Error function.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Error function evaluated at `a`.
    """


@dispatch
def sigmoid(a):
    """Sigmoid function.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Sigmoid function evaluated at `a`.
    """
    return 1 / (1 + exp(-a))


@dispatch
def softplus(a):
    """Softplus function.

    Args:
        a (tensor): Tensor.

    Returns:
        tensor: Softplus function evaluated at `a`.
    """
    zero = B.cast(B.dtype(a), 0)
    return log(1 + exp(-abs(a))) + maximum(a, zero)


@dispatch
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


@dispatch
@abstract(promote=2)
def add(a, b):  # pragma: no cover
    """Add two tensors.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor: Sum of `a` and `b`.
    """


@dispatch
@abstract(promote=2)
def subtract(a, b):  # pragma: no cover
    """Subtract two tensors.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor: `a` minus `b`.
    """


@dispatch
@abstract(promote=2)
def multiply(a, b):  # pragma: no cover
    """Multiply two tensors.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor: Product of `a` and `b`.
    """


@dispatch
@abstract(promote=2)
def divide(a, b):  # pragma: no cover
    """Divide two tensors.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor: `a` divided by `b`.
    """


@dispatch
@abstract(promote=2)
def power(a, power):  # pragma: no cover
    """Raise a tensor to a power.

    Args:
        a (tensor): Tensor.
        power (tensor): Power.

    Returns:
        tensor: `a` to the power of `power`.
    """


@dispatch
@abstract(promote=2)
def minimum(a, b):  # pragma: no cover
    """Take the minimum of two tensors.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor: Minimum of `a` and `b`.
    """


@dispatch
@abstract(promote=2)
def maximum(a, b):  # pragma: no cover
    """Take the maximum of two tensors.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor: Maximum of `a` and `b`.
    """


@dispatch
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


@dispatch
@abstract()
def min(a: Numeric, axis=None, squeeze=True):  # pragma: no cover
    """Take the minimum of a tensor, possibly along an axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Optional axis.
        squeeze (bool, optional): Squeeze the dimension after the reduction. Defaults
            to `True`.

    Returns:
        tensor: Reduced tensor.
    """


@dispatch
@abstract()
def argmin(a: Numeric, axis=None):  # pragma: no cover
    """Take the indices corresponding to the minimum of a tensor, possibly along an
    axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Optional axis.

    Returns:
        tensor: Indices.
    """


@dispatch
@abstract()
def max(a: Numeric, axis=None, squeeze=True):  # pragma: no cover
    """Take the maximum of a tensor, possibly along an axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Optional axis.
        squeeze (bool, optional): Squeeze the dimension after the reduction. Defaults
            to `True`.

    Returns:
        tensor: Reduced tensor.
    """


@dispatch
@abstract()
def argmax(a: Numeric, axis=None):  # pragma: no cover
    """Take the indices corresponding to the maximum of a tensor, possibly along an
    axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Optional axis.

    Returns:
        tensor: Indices.
    """


@dispatch
@abstract()
def sum(a: Numeric, axis=None, squeeze=True):  # pragma: no cover
    """Sum a tensor, possibly along an axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Optional axis.
        squeeze (bool, optional): Squeeze the dimension after the reduction. Defaults
            to `True`.

    Returns:
        tensor: Reduced tensor.
    """


@dispatch
def nansum(x, **kw_args):
    """Like :func:`sum`, but ignores `NaN`s."""
    available = ~B.isnan(x)
    x = B.where(available, x, B.zero(x))
    return B.sum(x, **kw_args)


@dispatch
@abstract()
def mean(a: Numeric, axis=None, squeeze=True):  # pragma: no cover
    """Take the mean of a tensor, possibly along an axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Optional axis.
        squeeze (bool, optional): Squeeze the dimension after the reduction. Defaults
            to `True`.

    Returns:
        tensor: Reduced tensor.
    """


@dispatch
def nanmean(x, **kw_args):
    """Like :func:`mean`, but ignores `NaN`s."""
    available = ~B.isnan(x)
    x = B.where(available, x, B.zero(x))
    # Need to cast `available` away from booleans for `sum` to work reliably.
    return B.sum(x, **kw_args) / B.sum(B.cast(B.dtype(x), available), **kw_args)


@dispatch
@abstract()
def std(a: Numeric, axis=None, squeeze=True):  # pragma: no cover
    """Compute the standard deviation of a tensor, possibly along an axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Optional axis.
        squeeze (bool, optional): Squeeze the dimension after the reduction. Defaults
            to `True`.

    Returns:
        tensor: Reduced tensor.
    """


@dispatch
def nanstd(x, **kw_args):
    """Like :func:`std`, but ignores `NaN`s."""
    inner_kw_args = kw_args.copy()
    inner_kw_args["squeeze"] = False
    mu = B.nanmean(x, **inner_kw_args)
    return B.sqrt(B.nanmean(B.power(x - mu, 2), **kw_args))


@dispatch
def logsumexp(a, axis=None, squeeze=True):  # pragma: no cover
    """Exponentiate a tensor, sum it, and then take the logarithm, possibly
    along an axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Optional axis.
        squeeze (bool, optional): Squeeze the dimension after the reduction. Defaults
            to `True`.

    Returns:
        tensor: Reduced tensor.
    """
    a_max = max(a, axis=axis, squeeze=True)
    # Expand `a_max` if an axis is specified.
    if axis is not None:
        a_max_inner = B.expand_dims(a_max, axis=axis)
    else:
        a_max_inner = a_max
    # Handle `squeeze` correctly.
    if axis is not None and not squeeze:
        a_max_outer = B.expand_dims(a_max, axis=axis)
    else:
        a_max_outer = a_max
    return log(sum(exp(a - a_max_inner), axis=axis, squeeze=squeeze)) + a_max_outer


# Logical reductions:


@dispatch
@abstract()
def all(a: Numeric, axis=None, squeeze=True):  # pragma: no cover
    """Logical all of a tensor, possibly along an axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Optional axis.
        squeeze (bool, optional): Squeeze the dimension after the reduction. Defaults
            to `True`.

    Returns:
        tensor: Reduced tensor.
    """


@dispatch
@abstract()
def any(a: Numeric, axis=None, squeeze=True):  # pragma: no cover
    """Logical any of a tensor, possibly along an axis.

    Args:
        a (tensor): Tensor.
        axis (int, optional): Optional axis.
        squeeze (bool, optional): Squeeze the dimension after the reduction. Defaults
            to `True`.

    Returns:
        tensor: Reduced tensor.
    """


# Logical comparisons:


@dispatch
@abstract(promote=2)
def lt(a, b):  # pragma: no cover
    """Check whether one tensor is strictly less than another.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor[bool]: `a` is strictly less than `b`.
    """


@dispatch
@abstract(promote=2)
def le(a, b):  # pragma: no cover
    """Check whether one tensor is less than or equal to another.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor[bool]: `a` is less than or equal to `b`.
    """


@dispatch
@abstract(promote=2)
def gt(a, b):  # pragma: no cover
    """Check whether one tensor is strictly greater than another.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor[bool]: `a` is strictly greater than `b`.
    """


@dispatch
@abstract(promote=2)
def ge(a, b):  # pragma: no cover
    """Check whether one tensor is greater than or equal to another.

    Args:
        a (tensor): First tensor.
        b (tensor): Second tensor.

    Returns:
        tensor[bool]: `a` is greater than or equal to `b`.
    """


@dispatch
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


@dispatch
def cond(condition: Numeric, f_true: FunctionType, f_false: FunctionType, *args):
    """An if-else statement that is part of the computation graph.

    Args:
        condition (bool): Condition to check.
        f_true (function): Function to execute if `condition` is true.
        f_false (function): Function to execute if `condition` is false.
        *args (object): Arguments to pass to `f_true` or `f_false` upon execution.
    """
    if control_flow.caching:
        control_flow.set_outcome("cond", condition, type=bool)
    elif control_flow.use_cache:
        if control_flow.get_outcome("cond"):
            return f_true(*args)
        else:
            return f_false(*args)
    return _cond(condition, f_true, f_false, *args)


@dispatch
def _cond(condition: Numeric, f_true: FunctionType, f_false: FunctionType, *args):
    if condition:
        return f_true(*args)
    else:
        return f_false(*args)


@dispatch
@abstract(promote=3)
def where(condition, a, b):  # pragma: no cover
    """In a broadcasting fashion, take from `x` if `condition` is true; otherwise,
    take from `y`.

    Args:
        condition (tensor): Condition.
        a (tensor): Tensor to take from if `condition` is true.
        b (tensor): Tensor to take from if `condition` is false.

    Returns:
        tensor: Tensor where every element is either from `x` or from `y`.
    """


@dispatch
def scan(f: Callable, xs, *init_state):
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

        state = convert(f(B.squeeze(state), xs[i]), tuple)
        new_state_shape = [B.shape(s) for s in state]

        # Check that the state shape remained constant.
        if new_state_shape != state_shape:
            raise RuntimeError(
                "Shape of state changed from {} to {}."
                "".format(state_shape, new_state_shape)
            )

        # Record the state, stacked over the various elements.
        states.append(B.stack(*state, axis=0))

    # Stack states over iterations.
    states = B.stack(*states, axis=0)

    # Put the elements dimension first and return.
    return B.transpose(states, perm=(1, 0) + tuple(range(2, B.rank(states))))


@dispatch
@abstract()
def sort(a: Numeric, axis=-1, descending=False):
    """Sort a tensor along an axis in ascending order.

    Args:
        a (tensor): Tensor to sort.
        axis (int, optional): Axis to sort along. Defaults to `-1`.
        descending (bool, optional): Sort in descending order. Defaults to
            `False`.

    Returns:
        tensor: `a`, but sorted.
    """


@dispatch
@abstract()
def argsort(a: Numeric, axis=-1, descending=False):
    """Get the indices that would a tensor along an axis in ascending order.

    Args:
        a (tensor): Tensor to sort.
        axis (int, optional): Axis to sort along. Defaults to `-1`.
        descending (bool, optional): Sort in descending order. Defaults to
            `False`.

    Returns:
        tensor: The indices that would sort `a`.
    """


@dispatch
@abstract(promote=2)
def quantile(a, q, axis=None):
    """Compute quantiles.

    Args:
        a (tensor): Tensor to compute quantiles of.
        q (tensor): Quantiles to compute. Must be numbers in `[0, 1]`.
        axis (int, optional): Axis to compute quantiles along. Defaults to `None`.

    Returns:
        tensor: Quantiles.
    """


NPOrNum = Union[NPNumeric, Number]  #: Type NumPy numeric or number.
add_conversion_method(AGNumeric, NPOrNum, lambda x: x._value)
add_conversion_method(TFNumeric, NPOrNum, lambda x: x.numpy())
add_conversion_method(TorchNumeric, NPOrNum, lambda x: x.detach().cpu().numpy())
add_conversion_method(JAXNumeric, NPOrNum, np.array)


@dispatch
def to_numpy(a):
    """Convert an object to NumPy.

    Args:
        a (object): Object to convert.

    Returns:
        `np.ndarray`: `a` as NumPy.
    """
    return convert(a, NPOrNum)


@dispatch
def to_numpy(*elements):
    return to_numpy(elements)


@dispatch
def to_numpy(a: list):
    return [to_numpy(x) for x in a]


@dispatch
def to_numpy(a: tuple):
    return tuple(to_numpy(x) for x in a)


@dispatch
def to_numpy(a: dict):
    return {k: to_numpy(v) for k, v in a.items()}
