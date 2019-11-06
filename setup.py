# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import subprocess

import numpy as np
from Cython.Build import build_ext
from setuptools import find_packages, setup, Extension

# Check that `gfortran` is available.
if subprocess.call(['which', 'gfortran']) != 0:
    raise RuntimeError('gfortran cannot be found. Please install gfortran. '
                       'On OS X, this can be done with "brew install gcc". '
                       'On Linux, "apt-get install gfortran" should suffice.')

# Compile TVPACK.
if subprocess.call('gfortran -fPIC -O2 '
                   '-c lab/bvn_cdf/tvpack.f '
                   '-o lab/bvn_cdf/tvpack.o', shell=True) != 0:
    raise RuntimeError('Compilation of TVPACK failed.')

# Default to use gcc as the compiler if `$CC` is not set.
if not 'CC' in os.environ or not os.environ['CC']:
    os.environ['CC'] = 'gcc'

# Ensure that `$CC` is not symlinked to `clang`, because the default shipped
# one often does not support OpenMP, but `gcc` does.
out = subprocess.check_output('$CC  --version', shell=True)
if 'clang' in out.decode('ascii'):
    # It is. Now try to find a `gcc` to replace it with.
    found = False
    for i in range(9, 3, -1):
        gcci = 'gcc-{}'.format(i)
        if subprocess.call(['which', gcci]) == 0:
            # Set both `$CC` and `$CXX` in this case, just to be sure.
            os.environ['CC'] = gcci
            os.environ['CXX'] = 'g++-{}'.format(i)
            found = True
            break

    # Ensure that one was found.
    if not found:
        raise RuntimeError('Your gcc runs clang, and no version of gcc could '
                           'be found. Please install gcc. On OS X, this can '
                           'be done with "brew install gcc".')

requirements = ['numpy>=1.16',
                'scipy>=1.3',

                'fdm',
                'plum-dispatch']

setup(packages=find_packages(exclude=['docs']),
      python_requires='>=3.6',
      install_requires=requirements,
      cmdclass={'build_ext': build_ext},
      ext_modules=[Extension('lab.bvn_cdf',
                             sources=['lab/bvn_cdf/bvn_cdf.pyx'],
                             include_dirs=[np.get_include()],
                             extra_compile_args=['-fPIC', '-O2', '-fopenmp'],
                             extra_objects=['lab/bvn_cdf/tvpack.o'],
                             extra_link_args=['-lgfortran', '-fopenmp'])],
      include_package_data=True)
