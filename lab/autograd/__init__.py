# noinspection PyUnresolvedReferences
from .. import *
from ..types import Number, NPNumeric, AGNumeric

Numeric = {Number, NPNumeric, AGNumeric}

from .generic import *
from .shaping import *
from .linear_algebra import *
from .random import *

# noinspection PyUnresolvedReferences
import autograd.numpy as anp
from ..types import _ag_retrievables

# Retrieve types.
for retrievable in _ag_retrievables:
    retrievable.retrieve()

# Alias to actual module.
sys.modules[__name__] = B
