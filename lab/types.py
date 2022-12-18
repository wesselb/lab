import sys

import numpy as np
from plum import (
    add_conversion_method,
    convert,
    add_promotion_rule,
    parametric,
    clear_all_cache,
    Union,
)
from plum.type import ResolvableType, ptype

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
    "_tf_retrievables",
    "_torch_retrievables",
    "_jax_retrievables",
]


@parametric
class UnimportedType:
    """Parametric type that represents a type that yet has to be loaded."""


class ModuleType(ResolvableType):
    """A type from a module that will resolve once it has been told that the
    module is imported.

    Args:
        module (str): Module.
        name (str): Name of type in the module.
    """

    def __init__(self, module, name):
        self.module = module
        self.name = name

        # Cache to hold the loaded type.
        self._type = None

        # Placeholder type to return whilst the type is unavailable.
        self._placeholder = UnimportedType[module + "." + name]

    def retrieve(self):
        """Retrieve the type, assuming that the module has been imported.

        Clears all cache after retrieval.
        """
        target = sys.modules[self.module]
        for part in self.name.split("."):
            target = getattr(target, part)
        self._type = ptype(target)
        clear_all_cache()

    def resolve(self):
        if self._type:
            return self._type
        else:
            return self._placeholder


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
_tf_retrievables = [
    _tf_tensor,
    _tf_indexedslices,
    _tf_variable,
    _tf_dtype,
    _tf_randomstate,
]

# Define PyTorch module types.
_torch_tensor = ModuleType("torch", "Tensor")
_torch_dtype = ModuleType("torch", "dtype")
_torch_device = ModuleType("torch", "device")
_torch_randomstate = ModuleType("torch", "Generator")
_torch_retrievables = [_torch_tensor, _torch_dtype, _torch_device, _torch_randomstate]

# Define AutoGrad module types.
_ag_tensor = ModuleType("autograd.tracer", "Box")
_ag_retrievables = [_ag_tensor]

# Define JAX module types.
if sys.version_info.minor > 7:  # jax>0.4 deprecated python-3.7 support, rely on older jax versions
    _jax_tensor = ModuleType("jaxlib.xla_extension", "ArrayImpl")
else:
    _jax_tensor = ModuleType("jax.interpreters.xla", "DeviceArray")
_jax_tracer = ModuleType("jax.core", "Tracer")
_jax_dtype = ModuleType("jax._src.numpy.lax_numpy", "_ScalarMeta")
_jax_device = ModuleType("jaxlib.xla_extension", "Device")
_jax_retrievables = [_jax_tensor, _jax_tracer, _jax_dtype, _jax_device]

# Numeric types:
Int = Union(*([int, Dimension] + np.sctypes["int"] + np.sctypes["uint"]), alias="Int")
Float = Union(*([float] + np.sctypes["float"]), alias="Float")
Complex = Union(*([complex] + np.sctypes["complex"]), alias="Complex")
Bool = Union(bool, np.bool_, alias="Bool")
Number = Union(Int, Bool, Float, Complex, alias="Number")
NPNumeric = Union(np.ndarray, alias="NPNumeric")
AGNumeric = Union(_ag_tensor, alias="AGNumeric")
TFNumeric = Union(_tf_tensor, _tf_variable, _tf_indexedslices, alias="TFNumeric")
TorchNumeric = Union(_torch_tensor, alias="TorchNumeric")
JAXNumeric = Union(_jax_tensor, _jax_tracer, alias="JAXNumeric")
Numeric = Union(
    Number, NPNumeric, AGNumeric, TFNumeric, JAXNumeric, TorchNumeric, alias="Numeric"
)

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
NPDType = Union(type, np.dtype, alias="NPDType")
AGDType = Union(NPDType, alias="AGDType")
TFDType = Union(_tf_dtype, alias="TFDType")
TorchDType = Union(_torch_dtype, alias="TorchDType")
JAXDType = Union(_jax_dtype, alias="JAXDType")
DType = Union(NPDType, TFDType, TorchDType, JAXDType, alias="DType")

# Create lookup for PyTorch data types that loads upon the first request.
_torch_lookup_cache = {}


def _torch_lookup(dtype):
    if not _torch_lookup_cache:
        # Cache is empty. Fill it.

        def _from_np(name):
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

        # `bool` can occur but isn't in `__all__`.
        for name in np.core.numerictypes.__all__ + ["bool"]:
            # Check that it is a type.
            if not isinstance(_from_np(name), type):
                continue

            # Attempt to get the PyTorch equivalent.
            try:
                _torch_lookup_cache[_module_attr("torch", name)] = _from_np(name)
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
    return _convert_back(getattr(np, "int" + "".join(name)), dtype)


@dispatch
def dtype_int(x):
    return dtype_int(dtype(x))


# Random state types:
NPRandomState = Union(np.random.RandomState, alias="NPRandomState")
AGRandomState = Union(NPRandomState, alias="AGRandomState")
TFRandomState = Union(_tf_randomstate, alias="TFRandomState")
TorchRandomState = Union(_torch_randomstate, alias="TorchRandomState")
JAXRandomState = Union(JAXNumeric, np.ndarray, alias="JAXRandomState")
RandomState = Union(
    NPRandomState,
    AGRandomState,
    TFRandomState,
    TorchRandomState,
    JAXRandomState,
    alias="RandomState",
)

# Device types:
TFDevice = Union(str, alias="TFDevice")
TorchDevice = Union(_torch_device, alias="TFDevice")
JAXDevice = Union(_jax_device, alias="JAXDevice")
Device = Union(TFDevice, TorchDevice, JAXDevice, alias="Device")

# Add conversions from non-string device types to strings.
add_conversion_method(TorchDevice, str, str)
add_conversion_method(JAXDevice, str, lambda d: f"{d.platform}:{d.id}")

# Framework types:
NP = Union(NPNumeric, NPDType, NPRandomState, alias="NP")
AG = Union(AGNumeric, AGDType, AGRandomState, alias="AG")
TF = Union(TFNumeric, TFDType, TFRandomState, TFDevice, alias="TF")
Torch = Union(TorchNumeric, TorchDType, TorchRandomState, TorchDevice, alias="Torch")
JAX = Union(JAXNumeric, JAXDType, JAXRandomState, JAXDevice, alias="JAX")
Framework = Union(NP, AG, TF, Torch, JAX, alias="Framework")
