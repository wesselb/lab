import jax.numpy as jnp
import jax.scipy.special as jsps
import jax.lax as lax
import jax
from types import FunctionType

from . import B, dispatch, Numeric
from .custom import jax_register
from ..custom import bvn_cdf, s_bvn_cdf
from ..types import JAXDType, JAXNumeric, NPNumeric, Number, Int

__all__ = []


@dispatch(Numeric)
def isnan(a):
    return jnp.isnan(a)


@dispatch(JAXNumeric)
def device(a):
    return str(a.device_buffer.device())


@dispatch(JAXNumeric)
def move_to_active_device(a):
    if B.Device.active_name:
        parts = B.Device.active_name.lower().split(":")
        if len(parts) == 1:
            return jax.device_put(a, jax.devices(parts[0])[0])
        elif len(parts) == 2:
            return jax.device_put(a, jax.devices(parts[0])[int(parts[1])])
        else:
            raise ValueError(
                f'Cannot parse device specification "{B.Device.active_name}".'
            )
    else:
        return a


@dispatch(JAXDType, [Int])
def zeros(dtype, *shape):
    return move_to_active_device(jnp.zeros(shape, dtype=dtype))


@dispatch(JAXDType, [Int])
def ones(dtype, *shape):
    return move_to_active_device(jnp.ones(shape, dtype=dtype))


@dispatch(JAXDType, Int, Int)
def _eye2(dtype, *shape):
    return move_to_active_device(jnp.eye(shape[0], shape[1], dtype=dtype))


@dispatch(JAXDType, object, object, Int)
def linspace(dtype, a, b, num):
    return move_to_active_device(jnp.linspace(a, b, num, dtype=dtype))


@dispatch(JAXDType, object, object, object)
def range(dtype, start, stop, step):
    return move_to_active_device(jnp.arange(start, stop, step, dtype=dtype))


@dispatch(JAXDType, JAXNumeric)
def cast(dtype, a):
    return a.astype(dtype)


@dispatch(JAXDType, {Number, NPNumeric})
def cast(dtype, a):
    return move_to_active_device(jnp.array(a, dtype=dtype))


@dispatch(Numeric)
def identity(a):
    # Do not return `a` identically.
    return 1 * a


@dispatch(Numeric)
def negative(a):
    return jnp.negative(a)


@dispatch(Numeric)
def abs(a):
    return jnp.abs(a)


@dispatch(Numeric)
def sign(a):
    return jnp.sign(a)


@dispatch(Numeric)
def sqrt(a):
    return jnp.sqrt(a)


@dispatch(Numeric)
def exp(a):
    return jnp.exp(a)


@dispatch(Numeric)
def log(a):
    return jnp.log(a)


@dispatch(Numeric)
def sin(a):
    return jnp.sin(a)


@dispatch(Numeric)
def cos(a):
    return jnp.cos(a)


@dispatch(Numeric)
def tan(a):
    return jnp.tan(a)


@dispatch(Numeric)
def tanh(a):
    return jnp.tanh(a)


@dispatch(Numeric)
def erf(a):
    return jsps.erf(a)


@dispatch(Numeric, Numeric)
def add(a, b):
    return jnp.add(a, b)


@dispatch(Numeric, Numeric)
def subtract(a, b):
    return jnp.subtract(a, b)


@dispatch(Numeric, Numeric)
def multiply(a, b):
    return jnp.multiply(a, b)


@dispatch(Numeric, Numeric)
def divide(a, b):
    return jnp.divide(a, b)


@dispatch(Numeric, Numeric)
def power(a, b):
    return jnp.power(a, b)


@dispatch(Numeric, Numeric)
def minimum(a, b):
    return jnp.minimum(a, b)


@dispatch(Numeric, Numeric)
def maximum(a, b):
    return jnp.maximum(a, b)


@dispatch(Numeric)
def min(a, axis=None):
    return jnp.min(a, axis=axis)


@dispatch(Numeric)
def max(a, axis=None):
    return jnp.max(a, axis=axis)


@dispatch(Numeric)
def sum(a, axis=None):
    return jnp.sum(a, axis=axis)


@dispatch(Numeric)
def mean(a, axis=None):
    return jnp.mean(a, axis=axis)


@dispatch(Numeric)
def std(a, axis=None):
    return jnp.std(a, axis=axis, ddof=0)


@dispatch(Numeric)
def all(a, axis=None):
    return jnp.all(a, axis=axis)


@dispatch(Numeric)
def any(a, axis=None):
    return jnp.any(a, axis=axis)


@dispatch(Numeric, Numeric)
def lt(a, b):
    return jnp.less(a, b)


@dispatch(Numeric, Numeric)
def le(a, b):
    return jnp.less_equal(a, b)


@dispatch(Numeric, Numeric)
def gt(a, b):
    return jnp.greater(a, b)


@dispatch(Numeric, Numeric)
def ge(a, b):
    return jnp.greater_equal(a, b)


f = jax_register(bvn_cdf, s_bvn_cdf)
dispatch(Numeric, Numeric, Numeric)(f)


@dispatch(Numeric, FunctionType, FunctionType, JAXNumeric, [JAXNumeric])
def _cond(condition, f_true, f_false, *xs):
    return lax.cond(condition, lambda xs_: f_true(*xs_), lambda xs_: f_false(*xs_), xs)


@dispatch(Numeric)
def sort(a, axis=-1, descending=False):
    if descending:
        return -jnp.sort(-a, axis=axis)
    else:
        return jnp.sort(a, axis=axis)


@dispatch(Numeric)
def argsort(a, axis=-1, descending=False):
    if descending:
        return jnp.argsort(-a, axis=axis)
    else:
        return jnp.argsort(a, axis=axis)
