# noinspection PyUnresolvedReferences
from .. import *
from .. import dispatch as dispatch_original
from ..shape import dispatch_unwrap_dimensions
from ..types import Number, TorchNumeric

dispatch = dispatch_unwrap_dimensions(dispatch_original)

from plum import Union

Numeric = Union[Number, TorchNumeric]

from .generic import *
from .shaping import *
from .linear_algebra import *
from .random import *

# noinspection PyUnresolvedReferences
from ..types import _torch_retrievables

# Retrieve types.
for retrievable in _torch_retrievables:
    retrievable.retrieve()

# Alias to actual module.
sys.modules[__name__] = B
