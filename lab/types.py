# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.variables import RefVariable
import torch
from plum import ListType, TupleType, Union, add_conversion_method, convert, \
    add_promotion_rule

from . import dispatch

__all__ = ['NPNumeric', 'TFNumeric', 'TorchNumeric', 'Numeric',
           'NPList', 'TFList', 'TorchList',
           'NPTuple', 'TFTuple', 'TorchTuple',
           'NPListOrTuple', 'TFListOrTuple', 'TorchListOrTuple', 'ListOrTuple',
           'NPShape', 'TFShape', 'TorchShape', 'Shape',
           'NPDType', 'TFDType', 'TorchDType', 'DType',
           'default_dtype', 'issubdtype',
           'NP', 'TF', 'Torch', 'Framework']

# Numeric types:

NPNumeric = np.ndarray
TFNumeric = Union(tf.Tensor, tf.Variable, RefVariable)
TorchNumeric = torch.Tensor
Numeric = Union(Number, NPNumeric, TFNumeric, TorchNumeric)

# Define promotion rules and the corresponding conversion methods.
add_promotion_rule(NPNumeric, TFNumeric, TFNumeric)
add_promotion_rule(NPNumeric, TorchNumeric, TorchNumeric)
add_promotion_rule(tf.Tensor, tf.Variable, TFNumeric)
add_conversion_method(NPNumeric, TFNumeric, tf.constant)
add_conversion_method(NPNumeric, TorchNumeric, torch.tensor)

# List types:

NPList = ListType(NPNumeric)
TFList = TFNumeric.expand(ListType)
TorchList = ListType(TorchNumeric)

# Tuple types:

NPTuple = TupleType(NPNumeric)
TFTuple = TFNumeric.expand(TupleType)
TorchTuple = TupleType(TorchNumeric)

# List or tuple types:

NPListOrTuple = Union(NPList, NPTuple)
TFListOrTuple = Union(TFList, TFTuple)
TorchListOrTuple = Union(TorchList, TorchTuple)
ListOrTuple = Union(list, tuple)

# Shape types:

NPShape = Union(ListType(int), ListType(Union()),
                TupleType(int), TupleType(Union()))
TFShape = Union(NPShape,
                tf.TensorShape,
                ListType(tf.Dimension),
                TupleType(tf.Dimension))
TorchShape = Union(NPShape, torch.Size)
Shape = Union(NPShape, TFShape, TorchShape)

# Data types:

NPDType = type(np.float64)
TFDType = tf.DType
TorchDType = torch.dtype
DType = Union(NPDType, TFDType, TorchDType)

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


# Framework types:

NP = Union(NPNumeric, NPListOrTuple, NPShape, NPDType)
TF = Union(TFNumeric, TFListOrTuple, TFShape, TFDType)
Torch = Union(TorchNumeric, TorchListOrTuple, TorchShape, TorchDType)
Framework = Union(NP, TF, Torch)

# Add conversion method for regular numbers.
add_conversion_method(Number, Framework, np.array)
