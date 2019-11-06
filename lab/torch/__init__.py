# noinspection PyUnresolvedReferences
from .. import *
from ..types import Number, TorchNumeric

Numeric = {Number, TorchNumeric}

from .generic import *
from .shaping import *
from .linear_algebra import *
from .random import *

# noinspection PyUnresolvedReferences
import torch
from ..types import _torch_retrievables

# Retrieve types.
for retrievable in _torch_retrievables:
    retrievable.retrieve()

# Alias to actual module.
sys.modules[__name__] = B
