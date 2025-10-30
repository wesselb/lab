from types import FunctionType
from typing import Union

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.scipy.special as jsps
from plum import isinstance

from ..custom import bvn_cdf, i_bvn_cdf, i_s_bvn_cdf, s_bvn_cdf
from ..types import (
    Int,
    JAXDType,
    JAXNumeric,
    JAXRandomState,
    NPNumeric,
    Number,
    _jax_tracer,
)
from . import B, Numeric, dispatch
from .custom import jax_register

__all__ = []


@dispatch
def isabstract(a: Numeric):
    return isinstance(a, _jax_tracer)


@dispatch
def _jit_run(
    f: FunctionType,
    compilation_cache: dict,
    jit_kw_args: dict,
    *args: Union[Numeric, JAXRandomState],
    **kw_args,
):
    if "jax" not in compilation_cache:
        # Run once to populate the control flow cache.
        f(*args, **kw_args)
        # Compile.
        compilation_cache["jax"] = jax.jit(f, **jit_kw_args)

    return compilation_cache["jax"](*args, **kw_args)


@dispatch
def isnan(a: Numeric):
    return jnp.isnan(a)


@dispatch
def real(a: Numeric):
    return jnp.real(a)


@dispatch
def imag(a: Numeric):
    return jnp.imag(a)


@dispatch
def device(a: JAXNumeric):
    devices = list(a.devices())
    if len(devices) != 1:
        raise RuntimeError("Could not determine device of JAX array.")
    return devices[0]


@dispatch
def to_active_device(a: JAXNumeric):
    if B.ActiveDevice.active_name:
        parts = B.ActiveDevice.active_name.lower().split(":")
        if len(parts) == 1:
            return jax.device_put(a, jax.devices(parts[0])[0])
        elif len(parts) == 2:
            return jax.device_put(a, jax.devices(parts[0])[int(parts[1])])
        else:
            raise ValueError(
                f'Cannot parse device specification "{B.ActiveDevice.active_name}".'
            )
    else:
        return a


@dispatch
def zeros(dtype: JAXDType, *shape: Int):
    return to_active_device(jnp.zeros(shape, dtype=dtype))


@dispatch
def ones(dtype: JAXDType, *shape: Int):
    return to_active_device(jnp.ones(shape, dtype=dtype))


@dispatch
def _eye2(dtype: JAXDType, *shape: Int):
    return to_active_device(jnp.eye(shape[0], shape[1], dtype=dtype))


@dispatch
def linspace(dtype: JAXDType, a, b, num: Int):
    return to_active_device(jnp.linspace(a, b, num, dtype=dtype))


@dispatch
def range(dtype: JAXDType, start, stop, step):
    return to_active_device(jnp.arange(start, stop, step, dtype=dtype))


@dispatch
def cast(dtype: JAXDType, a: JAXNumeric):
    return a.astype(dtype)


@dispatch
def cast(dtype: JAXDType, a: Union[Number, NPNumeric]):
    return to_active_device(jnp.array(a, dtype=dtype))


@dispatch
def identity(a: Numeric):
    # Do not return `a` identically.
    return jnp.multiply(1, a)


@dispatch
def round(a: Numeric):
    return jnp.round(a)


@dispatch
def floor(a: Numeric):
    return jnp.floor(a)


@dispatch
def ceil(a: Numeric):
    return jnp.ceil(a)


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
def log1p(a: Numeric):
    return jnp.log1p(a)


@dispatch
def sin(a: Numeric):
    return jnp.sin(a)


@dispatch
def arcsin(a: Numeric):
    return jnp.arcsin(a)


@dispatch
def cos(a: Numeric):
    return jnp.cos(a)


@dispatch
def arccos(a: Numeric):
    return jnp.arccos(a)


@dispatch
def tan(a: Numeric):
    return jnp.tan(a)


@dispatch
def arctan(a: Numeric):
    return jnp.arctan(a)


@dispatch
def tanh(a: Numeric):
    return jnp.tanh(a)


@dispatch
def arctanh(a: Numeric):
    return jnp.arctanh(a)


@dispatch
def loggamma(a: Numeric):
    return jsps.gammaln(a)


@dispatch
def erf(a: Numeric):
    return jsps.erf(a)


@dispatch
def softplus(a: JAXNumeric):
    return jnn.softplus(a)


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
def min(a: Numeric, axis: Union[Int, None] = None, squeeze: bool = True):
    return jnp.min(a, axis=axis, keepdims=not squeeze)


@dispatch
def argmin(a: Numeric, axis: Union[Int, None] = None):
    return jnp.argmin(a, axis=axis)


@dispatch
def max(a: Numeric, axis: Union[Int, None] = None, squeeze: bool = True):
    return jnp.max(a, axis=axis, keepdims=not squeeze)


@dispatch
def argmax(a: Numeric, axis: Union[Int, None] = None):
    return jnp.argmax(a, axis=axis)


@dispatch
def sum(a: Numeric, axis: Union[Int, None] = None, squeeze: bool = True):
    return jnp.sum(a, axis=axis, keepdims=not squeeze)


@dispatch
def prod(a: Numeric, axis: Union[Int, None] = None, squeeze: bool = True):
    return jnp.prod(a, axis=axis, keepdims=not squeeze)


@dispatch
def mean(a: Numeric, axis: Union[Int, None] = None, squeeze: bool = True):
    return jnp.mean(a, axis=axis, keepdims=not squeeze)


@dispatch
def std(a: Numeric, axis: Union[Int, None] = None, squeeze: bool = True):
    return jnp.std(a, axis=axis, ddof=0, keepdims=not squeeze)


@dispatch
def all(a: Numeric, axis: Union[Int, None] = None, squeeze: bool = True):
    return jnp.all(a, axis=axis, keepdims=not squeeze)


@dispatch
def any(a: Numeric, axis: Union[Int, None] = None, squeeze: bool = True):
    return jnp.any(a, axis=axis, keepdims=not squeeze)


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


@dispatch
def eq(a: Numeric, b: Numeric):
    return jnp.equal(a, b)


@dispatch
def ne(a: Numeric, b: Numeric):
    return jnp.not_equal(a, b)


_bvn_cdf = jax_register(bvn_cdf, i_bvn_cdf, s_bvn_cdf, i_s_bvn_cdf)


@dispatch
def bvn_cdf(a: Numeric, b: Numeric, c: Numeric):
    return _bvn_cdf(a, b, c)


@dispatch
def _cond(condition: JAXNumeric, f_true: FunctionType, f_false: FunctionType, *args):
    # We could use `jax.lax.cond` here, but that invokes compilation, which makes
    # repeated application of `B.cond` extremely slow.
    if condition:
        return f_true(*args)
    else:
        return f_false(*args)


@dispatch
def where(condition: Numeric, a: Numeric, b: Numeric):
    return jnp.where(condition, a, b)


@dispatch
def sort(a: Numeric, axis: Int = -1, descending: bool = False):
    if descending:
        return -jnp.sort(-a, axis=axis)
    else:
        return jnp.sort(a, axis=axis)


@dispatch
def argsort(a: Numeric, axis: Int = -1, descending: bool = False):
    if descending:
        return jnp.argsort(-a, axis=axis)
    else:
        return jnp.argsort(a, axis=axis)


@dispatch
def quantile(a: Numeric, q: Numeric, axis: Union[Int, None] = None):
    q = B.cast(B.dtype_float(q), q)  # JAX requires this to be a float.
    if jax.version.__version_info__ < (0, 8, 0):
        return jnp.quantile(a, q, axis=axis, interpolation="linear")
    else:
        return jnp.quantile(a, q, axis=axis, method="linear")
