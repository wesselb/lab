# noinspection PyUnresolvedReferences
from .. import *
from .. import dispatch as dispatch_original
from ..shape import dispatch_unwrap_dimensions
from ..types import Number, NPNumeric, JAXNumeric

dispatch = dispatch_unwrap_dimensions(dispatch_original)

from plum import Union

Numeric = Union[Number, NPNumeric, JAXNumeric]

from .generic import *
from .shaping import *
from .linear_algebra import *
from .random import *

# noinspection PyUnresolvedReferences
from ..types import _jax_retrievables

# Retrieve types.
for retrievable in _jax_retrievables:
    retrievable.retrieve()

# Alias to actual module.
sys.modules[__name__] = B
