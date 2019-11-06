# noinspection PyUnresolvedReferences
from .. import *
from .. import dispatch as dispatch_original
from ..types import Number, NPNumeric


# All methods here should have precedence, because NumPy forms the base of
# everything.
def dispatch(*args, **kw_args):
    kw_args['precedence'] = 1
    return dispatch_original(*args, **kw_args)


Numeric = {Number, NPNumeric}

from .generic import *
from .shaping import *
from .linear_algebra import *
from .random import *

# Alias to actual module.
sys.modules[__name__] = B
