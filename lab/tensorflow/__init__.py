# noinspection PyUnresolvedReferences
from .. import *
from .. import dispatch as dispatch_original
from ..shape import dispatch_unwrap_dimensions
from ..types import Number, NPNumeric, TFNumeric

dispatch = dispatch_unwrap_dimensions(dispatch_original)

Numeric = {Number, NPNumeric, TFNumeric}

from .generic import *
from .shaping import *
from .linear_algebra import *
from .random import *

# noinspection PyUnresolvedReferences
from ..types import _tf_retrievables

# Retrieve types.
for retrievable in _tf_retrievables:
    retrievable.retrieve()

# Alias to actual module.
sys.modules[__name__] = B
