# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np
import tensorflow as tf
import torch
from plum import add_promotion_rule, add_conversion_method

from .types import NP, TF, Torch, Numeric

__all__ = []

# Define promotion rules and the corresponding conversion methods.

add_promotion_rule(NP, TF, TF)
add_conversion_method(NP, TF, tf.constant)

add_promotion_rule(NP, Torch, Torch)
add_conversion_method(NP, Torch, torch.tensor)

# Add conversion method for regular numbers.
add_conversion_method(Number, Numeric, np.array)
