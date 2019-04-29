# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np
import tensorflow as tf
import torch
from plum import ListType, TupleType, Union

__all__ = ['NPNumeric', 'TFNumeric', 'TorchNumeric',
           'NPList', 'TFList', 'TorchList',
           'NPListOrTuple', 'TFListOrTuple', 'TorchListOrTuple',
           'NP', 'TF', 'Torch',
           'Framework', 'ListOrTuple', 'Numeric']

NPNumeric = np.ndarray  #: NumPy numeric type.
TFNumeric = Union(tf.Tensor, tf.Variable)  #: TensorFlow numeric type.
TorchNumeric = torch.Tensor  #: PyTorch numeric type.

NPList = ListType(NPNumeric)  #: NumPy list type.
# TODO: Use .expand_unions() in the below.
#: TensorFlow list type.
TFList = Union(ListType(tf.Tensor),
               ListType(tf.Variable),
               ListType(Union(tf.Tensor, tf.Variable)))
TorchList = ListType(TorchNumeric)  #: PyTorch list type.

NPTuple = TupleType(NPNumeric)  #: NumPy tuple type.
# TODO: Use .expand_unions() in the below.
#: TensorFlow tuple type.
TFTuple = Union(TupleType(tf.Tensor),
                TupleType(tf.Variable),
                TupleType(Union(tf.Tensor, tf.Variable)))
TorchTuple = TupleType(TorchNumeric)  #: PyTorch tuple type.

NPListOrTuple = Union(NPList, NPTuple)  #: NumPy list or tuple type.
TFListOrTuple = Union(TFList, TFTuple)  #: TensorFlow list or tuple type.
TorchListOrTuple = Union(TorchList, TorchTuple)  #: PyTorch list or tuple type.

NP = Union(NPNumeric, NPListOrTuple)  #: Everything accepted by NumPy type.
TF = Union(TFNumeric, TFListOrTuple)  #: Everything accepted by TensorFlow type.
#: Everything accepted by PyTorch type.
Torch = Union(TorchNumeric, TorchListOrTuple)

Framework = Union(NP, TF, Torch)  #: Everything accepted by frameworks type.
ListOrTuple = Union(list, tuple)  #: List or tuple.
Numeric = Union(Number,
                NPNumeric,
                TFNumeric,
                TorchNumeric)  #: Any accepted numeric type.
