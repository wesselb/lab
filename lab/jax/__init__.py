# noinspection PyUnresolvedReferences
from .. import *
from ..types import Number, NPNumeric, JaxNumeric

Numeric = {Number, NPNumeric, JaxNumeric}

from .generic import *
from .shaping import *
from .linear_algebra import *
from .random import *

# noinspection PyUnresolvedReferences
import autograd.numpy as anp
from ..types import _jax_retrievables

# Retrieve types.
for retrievable in _jax_retrievables:
    retrievable.retrieve()

# Alias to actual module.
sys.modules[__name__] = B
