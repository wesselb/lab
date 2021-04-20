from types import FunctionType

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy.special as jsps
from plum import Union

from . import B, dispatch, Numeric
from .custom import jax_register
from ..custom import bvn_cdf, s_bvn_cdf
from ..types import JAXDType, JAXNumeric, NPNumeric, Number, Int

__all__ = []


@dispatch
def isnan(a: Numeric):
    return jnp.isnan(a)


@dispatch
def device(a: JAXNumeric):
    return str(a.device_buffer.device())


@dispatch
def move_to_active_device(a: JAXNumeric):
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


@dispatch
def zeros(dtype: JAXDType, *shape: Int):
    return move_to_active_device(jnp.zeros(shape, dtype=dtype))


@dispatch
def ones(dtype: JAXDType, *shape: Int):
    return move_to_active_device(jnp.ones(shape, dtype=dtype))


@dispatch
def _eye2(dtype: JAXDType, *shape: Int):
    return move_to_active_device(jnp.eye(shape[0], shape[1], dtype=dtype))


@dispatch
def linspace(dtype: JAXDType, a, b, num: Int):
    return move_to_active_device(jnp.linspace(a, b, num, dtype=dtype))


@dispatch
def range(dtype: JAXDType, start, stop, step):
    return move_to_active_device(jnp.arange(start, stop, step, dtype=dtype))


@dispatch
def cast(dtype: JAXDType, a: JAXNumeric):
    return a.astype(dtype)


@dispatch
def cast(dtype: JAXDType, a: Union[Number, NPNumeric]):
    return move_to_active_device(jnp.array(a, dtype=dtype))


@dispatch
def identity(a: Numeric):
    # Do not return `a` identically.
    return 1 * a


@dispatch
def negative(a: Numeric):
    return jnp.negative(a)


@dispatch
def abs(a: Numeric):
    return jnp.abs(a)


@dispatch
def sign(a: Numeric):
    return jnp.sign(a)


@dispatch
def sqrt(a: Numeric):
    return jnp.sqrt(a)


@dispatch
def exp(a: Numeric):
    return jnp.exp(a)


@dispatch
def log(a: Numeric):
    return jnp.log(a)


@dispatch
def sin(a: Numeric):
    return jnp.sin(a)


@dispatch
def cos(a: Numeric):
    return jnp.cos(a)


@dispatch
def tan(a: Numeric):
    return jnp.tan(a)


@dispatch
def tanh(a: Numeric):
    return jnp.tanh(a)


@dispatch
def erf(a: Numeric):
    return jsps.erf(a)


@dispatch
def add(a: Numeric, b: Numeric):
    return jnp.add(a, b)


@dispatch
def subtract(a: Numeric, b: Numeric):
    return jnp.subtract(a, b)


@dispatch
def multiply(a: Numeric, b: Numeric):
    return jnp.multiply(a, b)


@dispatch
def divide(a: Numeric, b: Numeric):
    return jnp.divide(a, b)


@dispatch
def power(a: Numeric, b: Numeric):
    return jnp.power(a, b)


@dispatch
def minimum(a: Numeric, b: Numeric):
    return jnp.minimum(a, b)


@dispatch
def maximum(a: Numeric, b: Numeric):
    return jnp.maximum(a, b)


@dispatch
def min(a: Numeric, axis=None):
    return jnp.min(a, axis=axis)


@dispatch
def max(a: Numeric, axis=None):
    return jnp.max(a, axis=axis)


@dispatch
def sum(a: Numeric, axis=None):
    return jnp.sum(a, axis=axis)


@dispatch
def mean(a: Numeric, axis=None):
    return jnp.mean(a, axis=axis)


@dispatch
def std(a: Numeric, axis=None):
    return jnp.std(a, axis=axis, ddof=0)


@dispatch
def all(a: Numeric, axis=None):
    return jnp.all(a, axis=axis)


@dispatch
def any(a: Numeric, axis=None):
    return jnp.any(a, axis=axis)


@dispatch
def lt(a: Numeric, b: Numeric):
    return jnp.less(a, b)


@dispatch
def le(a: Numeric, b: Numeric):
    return jnp.less_equal(a, b)


@dispatch
def gt(a: Numeric, b: Numeric):
    return jnp.greater(a, b)


@dispatch
def ge(a: Numeric, b: Numeric):
    return jnp.greater_equal(a, b)


_bvn_cdf = jax_register(bvn_cdf, s_bvn_cdf)


@dispatch
def bvn_cdf(a: Numeric, b: Numeric, c: Numeric):
    return _bvn_cdf(a, b, c)


@dispatch
def _cond(
    condition: Numeric, f_true: FunctionType, f_false: FunctionType, *xs: JAXNumeric
):
    return lax.cond(condition, lambda xs_: f_true(*xs_), lambda xs_: f_false(*xs_), xs)


@dispatch
def sort(a: Numeric, axis=-1, descending=False):
    if descending:
        return -jnp.sort(-a, axis=axis)
    else:
        return jnp.sort(a, axis=axis)


@dispatch
def argsort(a: Numeric, axis=-1, descending=False):
    if descending:
        return jnp.argsort(-a, axis=axis)
    else:
        return jnp.argsort(a, axis=axis)
