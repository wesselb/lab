# noinspection PyUnresolvedReferences
from .. import *
from .. import dispatch as dispatch_original
from ..shape import dispatch_unwrap_dimensions
from ..types import JAXNumeric, NPNumeric, Number

dispatch = dispatch_unwrap_dimensions(dispatch_original)

from typing import Union

Numeric = Union[Number, NPNumeric, JAXNumeric]

import jax  # Load `jax` to load all new types.
from packaging.version import Version
from plum import clear_all_cache as _clear_all_cache

# In version before `0.5.1`, the type of JAX data types is located elsewhere.
if Version(jax.__version__) < Version("0.5.1"):  # pragma: no cover
    from ..types import _jax_dtype

    _jax_dtype._module = "jax._src.numpy.lax_numpy"


# noinspection PyUnresolvedReferences
from .generic import *
from .linear_algebra import *
from .random import *
from .shaping import *

# Clear cache to make sure that all newly loaded types are available.
_clear_all_cache()

# Alias to actual module.
sys.modules[__name__] = B
