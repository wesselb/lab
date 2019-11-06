import os
import sys

# Add package path.
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, '..')))

# noinspection PyUnresolvedReferences
import lab.torch
# noinspection PyUnresolvedReferences
import lab.tensorflow
