# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import sys

from plum import Dispatcher, Function, Tuple

__all__ = ['B']


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

    def _resolve_attr(self, name):
        for namespace in self.namespaces:
            try:
                return namespace, getattr(namespace, name)
            except AttributeError:
                continue
        raise AttributeError('Reference to \'{}\' not found.'.format(name))

    def __getattr__(self, name):
        namespace, attr = self._resolve_attr(name)

        # Check if `attr` is a property.
        if isinstance(attr, property):
            return attr.fget()

        # Ensure that `attr` is a Plum `Function` so that it can be
        # extended.
        if callable(attr) and not isinstance(attr, Function):
            # Make a function.
            f = Function(attr)

            # Set the fallback function to be the current function.
            f.register(Tuple([object]), attr)

            # Replace `attr` with a Plum `Function.
            setattr(namespace, name, f)
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
