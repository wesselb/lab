import sys
from typing import Union

import numpy as np
from plum import (
    ModuleType,
    activate_union_aliases,
    add_conversion_method,
    add_promotion_rule,
    convert,
    set_union_alias,
)

from . import dispatch
from .shape import Dimension

__all__ = [
    "Int",
    "Float",
    "Complex",
    "Bool",
    "Number",
    "NPNumeric",
    "AGNumeric",
    "TFNumeric",
    "TorchNumeric",
    "JAXNumeric",
    "Numeric",
    "NPDType",
    "AGDType",
    "TFDType",
    "TorchDType",
    "JAXDType",
    "DType",
    "NPRandomState",
    "AGRandomState",
    "TFRandomState",
    "TorchRandomState",
    "JAXRandomState",
    "RandomState",
    "TFDevice",
    "TorchDevice",
    "JAXDevice",
    "Device",
    "default_dtype",
    "dtype",
    "issubdtype",
    "promote_dtypes",
    "dtype_float",
    "dtype_int",
    "NP",
    "AG",
    "TF",
    "Torch",
    "JAX",
    "Framework",
]


def _module_call(module, method, *args, **kw_args):
    return getattr(sys.modules[module], method)(*args, **kw_args)


def _module_attr(module, attr):
    return getattr(sys.modules[module], attr)


# Define TensorFlow module types.
_tf_tensor = ModuleType("tensorflow", "Tensor")
_tf_indexedslices = ModuleType("tensorflow", "IndexedSlices")
_tf_variable = ModuleType("tensorflow", "Variable")
_tf_dtype = ModuleType("tensorflow", "DType")
_tf_randomstate = ModuleType("tensorflow", "random.Generator")

# Define PyTorch module types.
_torch_tensor = ModuleType("torch", "Tensor")
_torch_dtype = ModuleType("torch", "dtype")
_torch_device = ModuleType("torch", "device")
_torch_randomstate = ModuleType("torch", "Generator")

# Define AutoGrad module types.
_ag_tensor = ModuleType("autograd.tracer", "Box")

# Define JAX module types.
if sys.version_info.minor <= 7:  # pragma: specific no cover 3.8 3.9 3.10 3.11
    # `jax` 0.4 deprecated Python 3.7 support. Rely on older JAX versions.
    _jax_tensor = ModuleType("jax.interpreters.xla", "DeviceArray")
else:  # pragma: specific no cover 3.7
    _jax_tensor = ModuleType("jaxlib.xla_extension", "ArrayImpl")
_jax_tracer = ModuleType("jax.core", "Tracer")
_jax_dtype = ModuleType("jax._src.numpy.lax_numpy", "_ScalarMeta")
_jax_device = ModuleType("jaxlib.xla_extension", "Device")

# Numeric types:
Int = Union[tuple([int, Dimension] + np.sctypes["int"] + np.sctypes["uint"])]
Int = set_union_alias(Int, "B.Int")
Float = Union[tuple([float] + np.sctypes["float"])]
Float = set_union_alias(Float, "B.Float")
Complex = Union[tuple([complex] + np.sctypes["complex"])]
Complex = set_union_alias(Complex, "B.Complex")
Bool = Union[bool, np.bool_]
Bool = set_union_alias(Bool, "B.Bool")
Number = Union[Int, Bool, Float, Complex]
Number = set_union_alias(Number, "B.Number")
NPNumeric = Union[np.ndarray]
NPNumeric = set_union_alias(NPNumeric, "B.NPNumeric")
AGNumeric = Union[_ag_tensor]
AGNumeric = set_union_alias(AGNumeric, "B.AGNumeric")
TFNumeric = Union[_tf_tensor, _tf_variable, _tf_indexedslices]
TFNumeric = set_union_alias(TFNumeric, "B.TFNumeric")
TorchNumeric = Union[_torch_tensor]
TorchNumeric = set_union_alias(TorchNumeric, "B.TorchNumeric")
JAXNumeric = Union[_jax_tensor, _jax_tracer]
JAXNumeric = set_union_alias(JAXNumeric, "B.JAXNumeric")
Numeric = Union[
    Number,
    NPNumeric,
    AGNumeric,
    TFNumeric,
    JAXNumeric,
    TorchNumeric,
]
Numeric = set_union_alias(Numeric, "B.Numeric")

# Define corresponding promotion rules and conversion methods.
add_promotion_rule(NPNumeric, TFNumeric, TFNumeric)
add_promotion_rule(NPNumeric, TorchNumeric, TorchNumeric)
add_promotion_rule(NPNumeric, JAXNumeric, JAXNumeric)
add_promotion_rule(_tf_tensor, _tf_variable, TFNumeric)
add_conversion_method(
    NPNumeric, TFNumeric, lambda x: _module_call("tensorflow", "constant", x)
)
add_conversion_method(
    NPNumeric, TorchNumeric, lambda x: _module_call("torch", "tensor", x)
)
add_conversion_method(
    NPNumeric, JAXNumeric, lambda x: _module_call("jax.numpy", "asarray", x)
)

# Data types:
NPDType = Union[type, np.dtype]
NPDType = set_union_alias(NPDType, "B.NPDType")
AGDType = NPDType  # There is no specific data type for AutoGrad.
TFDType = Union[_tf_dtype]
TFDType = set_union_alias(TFDType, "B.TFDType")
TorchDType = Union[_torch_dtype]
TorchDType = set_union_alias(TorchDType, "B.TorchDType")
JAXDType = Union[_jax_dtype]
JAXDType = set_union_alias(JAXDType, "B.JAXDType")
DType = Union[NPDType, TFDType, TorchDType, JAXDType]
DType = set_union_alias(DType, "B.DType")

# Create lookup for PyTorch data types that loads upon the first request.
_torch_lookup_cache = {}


def _name_to_numpy_dtype(name):
    # We will want to get types from `np`, but the built-in types should be just
    # those.
    if name in {"int", "long"}:
        return int
    elif name == "bool":
        return bool
    elif name == "unicode":
        return str
    else:
        return getattr(np, name)


def _torch_lookup(dtype):
    if not _torch_lookup_cache:
        # Cache is empty. Fill it.

        # `bool` can occur but isn't in `__all__`.
        for name in np.core.numerictypes.__all__ + ["bool"]:
            _from_np = _name_to_numpy_dtype(name)

            # Check that it is a type.
            if not isinstance(_from_np, type):
                continue

            # Attempt to get the PyTorch equivalent.
            try:
                _torch_lookup_cache[_module_attr("torch", name)] = _from_np
            except AttributeError:
                # Could not find the PyTorch equivalent. That's okay.
                pass

    return _torch_lookup_cache[dtype]


def _name(x):
    try:
        return x.name
    except AttributeError:
        return x.__name__


# Add conversions between data types.
add_conversion_method(NPDType, TFDType, lambda x: _module_attr("tensorflow", _name(x)))
add_conversion_method(NPDType, TorchDType, lambda x: _module_attr("torch", _name(x)))
add_conversion_method(NPDType, JAXDType, lambda x: _module_attr("jax.numpy", _name(x)))
add_conversion_method(TorchDType, NPDType, _torch_lookup)
add_conversion_method(
    TorchDType, TFDType, lambda x: _module_attr("tensorflow", _name(_torch_lookup(x)))
)
add_conversion_method(
    TorchDType, JAXDType, lambda x: _module_attr("jax.numpy", _name(_torch_lookup(x)))
)
add_conversion_method(TFDType, NPDType, lambda x: x.as_numpy_dtype)
add_conversion_method(
    TFDType, TorchDType, lambda x: _module_attr("torch", _name(x.as_numpy_dtype))
)
add_conversion_method(
    TFDType, JAXDType, lambda x: _module_attr("jax.numpy", _name(x.as_numpy_dtype))
)
add_conversion_method(JAXDType, NPDType, lambda x: x.dtype.type)
add_conversion_method(
    JAXDType, TFDType, lambda x: _module_attr("tensorflow", _name(x.dtype))
)
add_conversion_method(JAXDType, TorchDType, lambda x: _module_attr("torch", _name(x)))

default_dtype = np.float64  #: Default dtype.


@dispatch
def dtype(a):
    """Determine the data type of an object.

    Args:
        a (tensor): Object to determine data type of.

    Returns:
        dtype: Data type of `a`.
    """
    return a.dtype


@dispatch
def dtype(a: Union[Number, NPNumeric]):
    try:
        # For the sake of consistency, return the underlying data type, and not a
        # `np.type`.
        return a.dtype.type
    except AttributeError:
        # It is likely a built-in. Its data type is then given by its type.
        return type(a)


@dispatch
def dtype(a: AGNumeric):
    # See above.
    return a.dtype.type


@dispatch
def dtype(a: JAXNumeric):
    # JAX gives NumPy data types back. Convert to JAX ones.
    return convert(a.dtype, JAXDType)


@dispatch
def dtype(*elements):
    return dtype(elements)


@dispatch
def dtype(elements: tuple):
    return promote_dtypes(*(dtype(a) for a in elements))


@dispatch
def issubdtype(dtype1: DType, dtype2: DType):
    """Check whether one data type is a subtype of another.

    Args:
        dtype1 (dtype): First data type.
        dtype2 (dtype): Second data type.

    Returns:
        bool: `dtype1` is a subtype of `dtype2`.
    """
    return np.issubdtype(convert(dtype1, NPDType), convert(dtype2, NPDType))


@dispatch
def promote_dtypes(first_dtype: DType, *dtypes: DType):
    """Find the smallest data type to which safely a number of the given data types can
    be cast.

    This function is sensitive to the order of the arguments. The result, however, is
    always valid.

    Args:
        *dtypes (dtype): Data types to promote. Must be at least one.

    Returns:
        dtype: Common data type. Will be of the type of the first given data type.
    """
    if len(dtypes) == 0:
        # There is just one data type given.
        return first_dtype
    # Perform promotion.
    common_dtype = np.promote_types(
        convert(first_dtype, NPDType), convert(dtypes[0], NPDType)
    )
    for dtype in dtypes[1:]:
        common_dtype = np.promote_types(common_dtype, convert(dtype, NPDType))
    return _convert_back(common_dtype.type, first_dtype)


@dispatch
def _convert_back():  # pragma: no cover
    pass


def _implement_convert_back(target):
    @dispatch
    def _convert_back(dtype: NPDType, _: target):
        return convert(dtype, target)


for target in [NPDType, AGDType, TFDType, TorchDType, JAXDType]:
    _implement_convert_back(target)


@dispatch
def dtype_float(dtype: DType):
    """Get the data type of an object and ensure that it is a floating type.

    Args:
        x (object): Data type or object to get data type of.

    Returns:
        dtype: Data type, but ensured to be floating.
    """
    return promote_dtypes(dtype, np.float16)


@dispatch
def dtype_float(x):
    return dtype_float(dtype(x))


@dispatch
def dtype_int(dtype: DType):
    """Get the data type of an object and get the integer equivalent.

    Args:
        x (object): Data type or object to get data type of.

    Returns:
        dtype: Data type, but ensured to be integer.
    """
    # TODO: Is there a better way of doing this?
    name = list(convert(dtype, NPDType).__name__)
    while name and name[0] not in set([str(i) for i in range(10)]):
        name.pop(0)
    return _convert_back(_name_to_numpy_dtype("int" + "".join(name)), dtype)


@dispatch
def dtype_int(x):
    return dtype_int(dtype(x))


# Random state types:
NPRandomState = Union[np.random.RandomState]
NPRandomState = set_union_alias(NPRandomState, "B.NPRandomState")
AGRandomState = NPRandomState  # There is no specific random state for AutoGrad.
TFRandomState = Union[_tf_randomstate]
TFRandomState = set_union_alias(TFRandomState, "B.TFRandomState")
TorchRandomState = Union[_torch_randomstate]
TorchRandomState = set_union_alias(TorchRandomState, "B.TorchRandomState")
JAXRandomState = Union[JAXNumeric, np.ndarray]
JAXRandomState = set_union_alias(JAXRandomState, "B.JAXRandomState")
RandomState = Union[
    NPRandomState,
    AGRandomState,
    TFRandomState,
    TorchRandomState,
    JAXRandomState,
]
RandomState = set_union_alias(RandomState, "B.RandomState")

# Device types:
TFDevice = Union[str]
TFDevice = set_union_alias(TFDevice, "B.TFDevice")
TorchDevice = Union[_torch_device]
TorchDevice = set_union_alias(TorchDevice, "B.TorchDevice")
JAXDevice = Union[_jax_device]
JAXDevice = set_union_alias(JAXDevice, "B.JAXDevice")
Device = Union[TFDevice, TorchDevice, JAXDevice]
Device = set_union_alias(Device, "B.Device")

# Add conversions from non-string device types to strings.
add_conversion_method(TorchDevice, str, str)
add_conversion_method(JAXDevice, str, lambda d: f"{d.platform}:{d.id}")

# Framework types:
NP = Union[NPNumeric, NPDType, NPRandomState]
NP = set_union_alias(NP, "B.NP")
AG = Union[AGNumeric, AGDType, AGRandomState]
AG = set_union_alias(AG, "B.AG")
TF = Union[TFNumeric, TFDType, TFRandomState, TFDevice]
TF = set_union_alias(TF, "B.TF")
Torch = Union[TorchNumeric, TorchDType, TorchRandomState, TorchDevice]
Torch = set_union_alias(Torch, "B.Torch")
JAX = Union[JAXNumeric, JAXDType, JAXRandomState, JAXDevice]
JAX = set_union_alias(JAX, "B.JAX")
Framework = Union[NP, AG, TF, Torch, JAX]
Framework = set_union_alias(Framework, "B.Framework")
