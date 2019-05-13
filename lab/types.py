# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import torch
from autograd.tracer import Box
from plum import Union, add_conversion_method, convert, add_promotion_rule
from tensorflow.python.ops.variables import RefVariable

from . import dispatch

__all__ = [
    'Int', 'Float', 'Bool', 'Number',
    'NPNumeric', 'TFNumeric', 'TorchNumeric', 'Numeric',
    'NPDimension', 'TorchDimension', 'TFDimension', 'Dimension',
    'NPDType', 'TFDType', 'TorchDType', 'DType',
    'default_dtype', 'issubdtype', 'dtype',
    'NP', 'TF', 'Torch', 'Framework']

# Numeric types:

Int = Union(*([int] + np.sctypes['int'] + np.sctypes['uint']), alias='Int')
Float = Union(*([float] + np.sctypes['float']), alias='Float')
Bool = Union(bool, np.bool_, alias='Bool')
Number = Union(Int, Float, alias='Number')
NPNumeric = Union(Number, Bool, np.ndarray, Box, alias='NPNumeric')
TFNumeric = Union(tf.Tensor, tf.Variable, RefVariable, alias='TFNumeric')
TorchNumeric = Union(torch.Tensor, alias='TorchNumeric')
Numeric = Union(NPNumeric, TFNumeric, TorchNumeric, alias='Numeric')

# Define promotion rules and the corresponding conversion methods.
add_promotion_rule(NPNumeric, TFNumeric, TFNumeric)
add_promotion_rule(NPNumeric, TorchNumeric, TorchNumeric)
add_promotion_rule(tf.Tensor, tf.Variable, TFNumeric)
add_conversion_method(NPNumeric, TFNumeric, tf.constant)
add_conversion_method(NPNumeric, TorchNumeric, torch.tensor)

# Dimension types:

NPDimension = Union(int, alias='NPDimension')
TFDimension = Union(NPDimension, tf.Dimension, alias='TFDimension')
TorchDimension = Union(NPDimension, alias='TorchDimension')
Dimension = Union(NPDimension, TFDimension, TorchDimension, alias='Dimension')

# Define corresponding conversion methods.
add_conversion_method(tf.Dimension, int, lambda x: x.value)

# Data types:

NPDType = Union(type(np.float64), alias='NPDType')
TFDType = Union(tf.DType, alias='TFDType')
TorchDType = Union(torch.dtype, alias='TorchDType')
DType = Union(NPDType, TFDType, TorchDType, alias='DType')

# Create lookup for PyTorch dtypes.
_torch_to_np_lookup = {}
for name in np.core.numerictypes.__all__:
    # Check that it is a type.
    if not isinstance(getattr(np, name), type):
        continue

    # Attempt to get the PyTorch equivalent.
    try:
        _torch_to_np_lookup[getattr(torch, name)] = getattr(np, name)
    except AttributeError:
        # Could not find the PyTorch equivalent. That's okay.
        pass

# Add conversions between data types.
add_conversion_method(NPDType, TFDType, lambda x: getattr(tf, x.__name__))
add_conversion_method(NPDType, TorchDType, lambda x: getattr(torch, x.__name__))
add_conversion_method(TorchDType, NPDType, lambda x: _torch_to_np_lookup[x])
add_conversion_method(TorchDType, TFDType,
                      lambda x: getattr(tf, _torch_to_np_lookup[x].__name__))
add_conversion_method(TFDType, NPDType, lambda x: x.as_numpy_dtype)
add_conversion_method(TFDType, TorchDType,
                      lambda x: getattr(torch, x.as_numpy_dtype.__name__))

# NumPy data types can be of type `np.dtype`. Convert to the proper types.
add_conversion_method(np.dtype, DType, lambda x: x.type)

default_dtype = np.float64  #: Default dtype.


@dispatch(type, type)
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
    return issubdtype(convert(dtype1, type), convert(dtype2, type))


@dispatch(object)
def dtype(a):
    """Determine the data type of an object.

    Args:
        a (tensor): Object to determine data type of.
    """
    if hasattr(a, 'dtype'):
        return convert(a.dtype, DType)
    else:
        return type(a)


# Framework types:

NP = Union(NPNumeric, NPDType, alias='NP')
TF = Union(TFNumeric, TFDType, alias='TF')
Torch = Union(TorchNumeric, TorchDType, alias='Torch')
Framework = Union(NP, TF, Torch, alias='Framework')
