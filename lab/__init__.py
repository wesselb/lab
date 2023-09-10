import sys

from plum import Dispatcher

B = sys.modules[__name__]  # Allow both import styles.
dispatch = Dispatcher()  # This dispatch namespace will be used everywhere.

from .control_flow import *
from .generic import *
from .linear_algebra import *
from .numpy import *
from .random import *
from .shaping import *
from .types import *

# Fix namespace issues with `B.bvn_cdf` simply by setting it explicitly.
B.bvn_cdf = B.generic.bvn_cdf
