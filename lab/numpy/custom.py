# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from autograd.extend import primitive, defvjp_argnums

__all__ = ['autograd_register']


def autograd_register(f, s_f):
    """Register a function and its sensitivity for AutoGrad.

    Args:
        f (function): Function to register.
        s_f (function): Sensitivity of `f`.

    Returns:
        function: AutoGrad primitive.
    """
    # Create a primitive for `f`.
    f_primitive = primitive(f)

    # Register the sensitivity.
    def vjp_argnums(nums, y, args, kw_args):
        def vjp(s_y):
            return tuple(np.take(s_f(s_y, y, *args, **kw_args), nums))

        return vjp

    defvjp_argnums(f_primitive, vjp_argnums)

    # Return the AutoGrad primitive.
    return f_primitive
