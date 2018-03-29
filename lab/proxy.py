# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import inspect
import sys

__all__ = ['B']


def default_args(f, **def_args):
    """Change default values for keyword arguments of a function.

    Omitting the argument or setting the keyword to `None` will yield the
    newly-specified default.

    Args:
        f (function): Function under consideration.
        **def_args: Keys correspond to names of keyword arguments, and values
            correspond to new default values.
    """
    # Get the names of all the arguments of `f`.
    try:
        try:
            # Python 3:
            args = inspect.signature(f).parameters
        except AttributeError:
            # Python 2:
            args, _, _, _ = inspect.getargspec(f)
    except TypeError:
        return f

    # Filter any default arguments that not apply to `f`.
    def_args = {k: def_args[k] for k in set(args) & set(def_args.keys())}

    def wrapped_f(*args, **kw_args):
        for k, v in def_args.items():
            if k not in kw_args or kw_args[k] is None:
                kw_args[k] = v
        return f(*args, **kw_args)

    return wrapped_f


class Proxy(object):
    """A generic proxy adhering to NumPy's interface as much as possible.

    Args:
        module (module): This module.
    """

    _default_kw_args = {}

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
        object.__setattr__(self, '_namespaces', namespaces_list)

    def _resolve_attr(self, name):
        for namespace in self.namespaces:
            if hasattr(namespace, name):
                return namespace, getattr(namespace, name)
        raise AttributeError('reference to \'{}\' not found.'.format(name))

    def __getattr__(self, name):
        _, attr = self._resolve_attr(name)
        if isinstance(attr, property):
            return attr.fget()
        elif callable(attr):
            return self._module.default_args(attr, **self._default_kw_args)
        else:
            return attr

    def __setattr__(self, name, value):
        namespace, _ = self._resolve_attr(name)
        setattr(namespace, name, value)

    def backend_to_tf(self):
        """Switch the backend to TensorFlow."""
        from . import generic as gen
        from .proxies import tensorflow as tf
        self.set_namespaces([gen, tf])

    def backend_to_np(self):
        """Switch the backend to NumPy."""
        from . import generic as gen
        from .proxies import numpy as np
        self.set_namespaces([gen, np])

    def backend_to_torch(self):
        """Switch the backend to PyTorch."""
        from . import generic as gen
        from .proxies import pytorch as torch
        self.set_namespaces([gen, torch])

    def set_default_dtype(self, dtype):
        """Set the default data type.

        Args:
            dtype (data type): New default data type.
        """
        self._default_kw_args['dtype'] = dtype


B = Proxy(sys.modules[__name__])
