import sys

import numpy as np
from plum import (
    Union,
    add_conversion_method,
    convert,
    add_promotion_rule,
    ResolvableType,
    as_type,
    parametric,
    clear_all_cache
)

from . import dispatch

__all__ = ['Int',
           'Float',
           'Bool',
           'Number',

           'NPNumeric',
           'AGNumeric',
           'TFNumeric',
           'TorchNumeric',
           'Numeric',

           'NPDType',
           'AGDType',
           'TFDType',
           'TorchDType',
           'DType',

           'default_dtype',
           'issubdtype',
           'dtype',

           'NP',
           'AG',
           'TF',
           'Torch',
           'Framework',

           '_tf_retrievables',
           '_torch_retrievables']


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
        self._placeholder = UnimportedType(module + '.' + name)

    def retrieve(self):
        """Retrieve the type, assuming that the module has been imported.

        Clears all cache after retrieval.
        """
        self._type = as_type(getattr(sys.modules[self.module], self.name))
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
_tf_tensor = ModuleType('tensorflow', 'Tensor')
_tf_indexedslices = ModuleType('tensorflow', 'IndexedSlices')
_tf_variable = ModuleType('tensorflow', 'Variable')
_tf_dtype = ModuleType('tensorflow', 'DType')
_tf_retrievables = [_tf_tensor, _tf_indexedslices, _tf_variable, _tf_dtype]

# Define PyTorch module types.
_torch_tensor = ModuleType('torch', 'Tensor')
_torch_dtype = ModuleType('torch', 'dtype')
_torch_retrievables = [_torch_tensor, _torch_dtype]

# Define AutoGrad module types.
_ag_tensor = ModuleType('autograd.tracer', 'Box')
_ag_retrievables = [_ag_tensor]

# Numeric types:
Int = Union(*([int] + np.sctypes['int'] + np.sctypes['uint']), alias='Int')
Float = Union(*([float] + np.sctypes['float']), alias='Float')
Bool = Union(bool, np.bool_, alias='Bool')
Number = Union(Int, Bool, Float, alias='Number')
NPNumeric = Union(np.ndarray, alias='NPNumeric')
AGNumeric = Union(_ag_tensor, alias='AGNumeric')
TFNumeric = Union(_tf_tensor,
                  _tf_variable,
                  _tf_indexedslices, alias='TFNumeric')
TorchNumeric = Union(_torch_tensor, alias='TorchNumeric')
Numeric = Union(Number,
                NPNumeric,
                AGNumeric,
                TFNumeric,
                TorchNumeric, alias='Numeric')

# Define corresponding promotion rules and conversion methods.
add_promotion_rule(NPNumeric, TFNumeric, TFNumeric)
add_promotion_rule(NPNumeric, TorchNumeric, TorchNumeric)
add_promotion_rule(_tf_tensor, _tf_variable, TFNumeric)
add_conversion_method(NPNumeric, TFNumeric,
                      lambda x: _module_call('tensorflow', 'constant', x))
add_conversion_method(NPNumeric, TorchNumeric,
                      lambda x: _module_call('torch', 'tensor', x))

# Data types:
NPDType = Union(type, np.dtype, alias='NPDType')
AGDType = Union(NPDType, alias='AGDType')
TFDType = Union(_tf_dtype, alias='TFDType')
TorchDType = Union(_torch_dtype, alias='TorchDType')
DType = Union(NPDType, TFDType, TorchDType, alias='DType')

# Create lookup for PyTorch data types that loads upon the first request.
_torch_lookup_cache = {}


def _torch_lookup(dtype):
    if not _torch_lookup_cache:
        # Cache is empty. Fill it.

        for name in np.core.numerictypes.__all__:
            # Check that it is a type.
            if not isinstance(getattr(np, name), type):
                continue

            # Attempt to get the PyTorch equivalent.
            try:
                _torch_lookup_cache[_module_attr('torch', name)] = \
                    getattr(np, name)
            except AttributeError:
                # Could not find the PyTorch equivalent. That's okay.
                pass

    return _torch_lookup_cache[dtype]


# Add conversions between data types.
def _name(x):
    try:
        return x.name
    except AttributeError:
        return x.__name__


add_conversion_method(NPDType, TFDType,
                      lambda x: _module_attr('tensorflow', _name(x)))
add_conversion_method(NPDType, TorchDType,
                      lambda x: _module_attr('torch', _name(x)))
add_conversion_method(TorchDType, NPDType, _torch_lookup)
add_conversion_method(TorchDType, TFDType,
                      lambda x: _module_attr('tensorflow',
                                             _name(_torch_lookup(x))))
add_conversion_method(TFDType, NPDType, lambda x: x.as_numpy_dtype)
add_conversion_method(TFDType, TorchDType,
                      lambda x: _module_attr('torch',
                                             _name(x.as_numpy_dtype)))

default_dtype = np.float64  #: Default dtype.


@dispatch(NPDType, NPDType)
def issubdtype(dtype1, dtype2):
    """Check whether one data type is a subtype of another.

    Args:
        dtype1 (dtype): First data type.
        dtype2 (dtype): Second data type.

    Returns:
        bool: `dtype1` is a subtype of `dtype2`.
    """
    return np.issubdtype(dtype1, dtype2)


@dispatch(object, object)
def issubdtype(dtype1, dtype2):
    return issubdtype(convert(dtype1, NPDType), convert(dtype2, NPDType))


@dispatch(object)
def dtype(a):
    """Determine the data type of an object.

    Args:
        a (tensor): Object to determine data type of.
    """
    try:
        return a.dtype
    except AttributeError:
        return type(a)


# Framework types:
NP = Union(NPNumeric, NPDType, alias='NP')
AG = Union(AGNumeric, AGDType, alias='AG')
TF = Union(TFNumeric, TFDType, alias='TF')
Torch = Union(TorchNumeric, TorchDType, alias='Torch')
Framework = Union(NP, TF, Torch, alias='Framework')
