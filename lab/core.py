# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

__all__ = ['say_hi']
log = logging.getLogger(__name__)


def say_hi():
    """Say hi to the world."""
    print('Hi!')