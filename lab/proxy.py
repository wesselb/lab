# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import sys
from functools import wraps

from plum import Dispatcher, Function, Tuple

from .util import Namespace

__all__ = ['B']

extensions = Namespace()
"""A namespace where extensions of functions will be put."""


class Proxy(object):
    """A generic proxy adhering to NumPy's interface as much as possible.

    Args:
        module (module): This module.
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
        for namespace in namespaces:
            try:
                return namespace, getattr(namespace, name)
            except AttributeError:
                continue

        if with_extensions:
            # Reference could not be found. Create an empty Plum function that
            # can be extended.
            def function():
                """Automatically generated function."""
                pass

            function.__name__ = name

            # Save the function and return.
            setattr(extensions, name, Function(f=function))
            return extensions, getattr(extensions, name)
        else:
            # Reference could not be found, but extensions are not enabled.
            # Throw an error.
            raise AttributeError('Reference to \'{}\' not found.'.format(name))

    def __getattr__(self, name):
        namespace, attr = self._resolve_attr(name)

        # Check if `attr` is a property.
        if isinstance(attr, property):
            return attr.fget()

        # Ensure that `attr` is a Plum `Function` so that it can be
        # extended.
        if callable(attr) and not isinstance(attr, Function):
            # Create the function to be wrapped. This function needs to lookup
            # `attr` again, because the backend could've changed.
            # TODO: Dynamically change `@wraps(attr)` if backend is changed.
            @wraps(attr)
            def lookup_attr(*args, **kw_args):
                # Directly call `_resolve_attr` and ignore extensions.
                found_attr = self._resolve_attr(name, with_extensions=False)[1]
                return found_attr(*args, **kw_args)

            # Make a Plum function.
            f = Function(lookup_attr)

            # Set the fallback function to be the current function.
            f.register(Tuple([object]), lookup_attr)

            # Replace `attr` with a Plum `Function` by attaching it to the
            # extensions namespace.
            setattr(extensions, name, f)
            attr = f

        return attr

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
