# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import sys

from plum import Dispatcher

B = sys.modules[__name__]  # Allow both import styles.
dispatch = Dispatcher()  # This dispatch namespace will be used everywhere.

from .generic import *
from .shaping import *
from .linear_algebra import *
from .random import *

from .autograd import *

from .types import *

# Fix namespace issues with `B.bvn_cdf` simply by setting it explicitly.
B.bvn_cdf = B.generic.bvn_cdf
