# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np
import tensorflow as tf
import torch
from plum import Union

__all__ = ['NP', 'TF', 'Torch', 'Numeric', 'Framework']

NP = np.ndarray  #: NumPy type.
TF = Union(tf.Tensor, tf.Variable)  #: TensorFlow type.
Torch = torch.Tensor  #: PyTorch type.

Framework = Union(NP, TF, Torch)  #: Framework-only numeric type.
Numeric = Union(Number, Framework)  #: Numeric type.
