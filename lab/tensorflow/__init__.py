# noinspection PyUnresolvedReferences
from .. import *
from ..types import Number, NPNumeric, TFNumeric

Numeric = {Number, NPNumeric, TFNumeric}

from .generic import *
from .shaping import *
from .linear_algebra import *
from .random import *

# noinspection PyUnresolvedReferences
import tensorflow as tf
from ..types import _tf_retrievables

# Retrieve types.
for retrievable in _tf_retrievables:
    retrievable.retrieve()

# Alias to actual module.
sys.modules[__name__] = B
