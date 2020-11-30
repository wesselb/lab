import os
import sys

# Add package path.
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, "..")))

# noinspection PyUnresolvedReferences
import lab.autograd

# noinspection PyUnresolvedReferences
import lab.torch

# noinspection PyUnresolvedReferences
import lab.tensorflow

# noinspection PyUnresolvedReferences
import lab.jax

import jax


# We need `float64`s for testing.
jax.config.update("jax_enable_x64", True)
