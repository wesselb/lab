# noinspection PyUnresolvedReferences
from .. import *
from .. import dispatch as dispatch_original
from ..shape import dispatch_unwrap_dimensions
from ..types import NPNumeric, Number

# All methods here should have precedence, because NumPy forms the base of everything.
dispatch_original = dispatch_original(precedence=1)

dispatch = dispatch_unwrap_dimensions(dispatch_original)

from typing import Union

Numeric = Union[Number, NPNumeric]

from .generic import *
from .linear_algebra import *
from .random import *
from .shaping import *

# Alias to actual module.
sys.modules[__name__] = B
