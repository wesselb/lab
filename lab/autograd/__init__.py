# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

# noinspection PyUnresolvedReferences
from .. import *

from .generic import *
from .shaping import *
from .linear_algebra import *
from .random import *

# Alias to actual module.
sys.modules[__name__] = B
