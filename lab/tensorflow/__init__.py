# noinspection PyUnresolvedReferences
from .. import *
from .. import dispatch as dispatch_original
from ..shape import dispatch_unwrap_dimensions
from ..types import Number, NPNumeric, TFNumeric

dispatch = dispatch_unwrap_dimensions(dispatch_original)

from plum import Union

Numeric = Union[Number, NPNumeric, TFNumeric]

from .generic import *
from .shaping import *
from .linear_algebra import *
from .random import *

import tensorflow as tf

# noinspection PyUnresolvedReferences
from ..types import _tf_retrievables

# Retrieve types.
for retrievable in _tf_retrievables:
    retrievable.retrieve()

# Set TF device manager.
B.Device._tf_manager = tf.device

# Alias to actual module.
sys.modules[__name__] = B
