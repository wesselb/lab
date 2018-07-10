# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
import sys
from types import FunctionType

from plum import Dispatcher, Function, Tuple

from .util import Namespace

__all__ = ['B']

log = logging.getLogger(__name__)

extensions = Namespace()
"""A namespace where extensions of functions will be put."""

Accepted = {type, type(None), int, float, list, tuple, set, bool, str,
            FunctionType}
"""A list of accepted types.

The numeric type and data type type will be added dynamically to this set.
"""


class Proxy(object):
    """A generic proxy adhering to NumPy's interface as much as possible.

    Args:
        module (module): This module.
    """
    _privileged = ['promote', 'convert', 'add_promotion_rule']
    """Privileged attributes.
    
    These are guaranteed to be found and will be returned without any wrapping.
    """

    def __init__(self, module):
        object.__setattr__(self, '_module', module)

    @property
    def namespaces(self):
        """List of namespaces which contain implementations.

        Due to the circular dependency of the proxy and implementations,
        the implementations cannot be loaded creation of this object, since
        otherwise this object, which defines the proxy namespace, will not
        exist.
        """
        if not hasattr(self, '_namespaces'):
            self.backend_to_np()
        return self._namespaces

    def set_namespaces(self, namespaces_list):
        """Set the namespaces which contain implementations.

        Args:
            namespaces_list (list): List of namespaces. The proxy will try
                the namespaces in the provided order.
        """
        # Clear all dispatch cache before changing.
        Dispatcher.clear_all_cache()
        # Change namespaces.
        object.__setattr__(self, '_namespaces', namespaces_list)

    def _resolve_attr(self, name, with_extensions=True):
        namespaces = self.namespaces

        # Add the extensions namespace if required.
        if with_extensions:
            namespaces = [extensions] + namespaces

        # Walk through the namespaces and attempt to get the function.
        namespace, attr = None, None
        for namespace in namespaces:
            if hasattr(namespace, name):
                attr = getattr(namespace, name)
                break

        if name in Proxy._privileged:
            # The attribute is privileged. Return without any fancy wrapping.
            # Privileged attributes are guaranteed to be found.
            return namespace, attr

        elif attr and not callable(attr):
            # If the attribute is found, but not callable, simply return and
            # avoid any fancy extensions.
            return namespace, attr

        elif with_extensions and namespace != extensions:
            # Extensions are enabled, but the resolved namespace is not
            # `extensions`. Create a Plum function that can globally be
            # extended.
            def function():
                """Automatically generated function."""
                pass

            function.__name__ = name

            # Create Plum function.
            f = Function(f=function)

            # If the function was found, then set the method for acceptable
            # types to the one found and add a promotion method.
            if namespace:
                def lookup_attr(*args, **kw_args):
                    # Lookup `attr`, ignoring extensions. This should always
                    # exist.
                    _, found_attr = self._resolve_attr(name,
                                                       with_extensions=False)
                    return found_attr(*args, **kw_args)

                # Set the precedence of this method to `-1` to favour
                # user-defined methods.
                f.register(Tuple([{B.Numeric, B.DType} | Accepted]),
                           lookup_attr,
                           precedence=-1)

                def promote_arguments(*args, **kw_args):
                    promoted_args = B.promote(*args)

                    # If promotion did not change the types, we'll end up here
                    # again. Then simply lookup the attribute.
                    if all([type(x) == type(y)
                            for x, y in zip(promoted_args, args)]):
                        return lookup_attr(*args, **kw_args)

                    # Promotion did something. Cool!
                    return getattr(B, name)(*promoted_args, **kw_args)

                f.register(Tuple([object]), promote_arguments)

            # Save the created function and return.
            setattr(extensions, name, f)
            return extensions, f

        elif attr:
            # Successfully found callable attribute. Return.
            return namespace, attr

        else:
            # Reference could not be found. Throw an error.
            raise AttributeError('Reference to \'{}\' not found.'.format(name))

    def __getattr__(self, name):
        namespace, attr = self._resolve_attr(name)
        return attr.fget() if isinstance(attr, property) else attr

    def __setattr__(self, name, value):
        namespace, _ = self._resolve_attr(name)
        setattr(namespace, name, value)

    def __dir__(self):
        cur_dir = set(object.__dir__(self))
        for namespace in self._namespaces:
            cur_dir |= set(dir(namespace))
        return sorted(cur_dir)

    def backend_to_tf(self):
        """Switch the backend to TensorFlow."""
        from . import generic as gen
        from .proxies import tensorflow as tf_proxy
        import tensorflow as tf
        self.set_namespaces([tf_proxy, gen, tf])

    def backend_to_np(self):
        """Switch the backend to NumPy."""
        from . import generic as gen
        from .proxies import numpy as np_proxy
        import autograd.numpy as np
        self.set_namespaces([np_proxy, gen, np])

    def backend_to_torch(self):
        """Switch the backend to PyTorch."""
        from . import generic as gen
        from .proxies import torch as torch_proxy
        import torch
        self.set_namespaces([torch_proxy, gen, torch])


B = Proxy(sys.modules[__name__])
