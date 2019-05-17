# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os

import numpy as np
from Cython.Build import build_ext
from setuptools import find_packages, setup, Extension

with open('README.md') as f:
    readme = f.read()

with open('LICENCE') as f:
    licence = f.read()

# Compile TVPACK.
command = 'gfortran -fPIC -O2 -c lab/bvn_cdf/tvpack.f -o lab/bvn_cdf/tvpack.o'
print(command)
os.system(command)

# Use gcc as the compiler.
os.environ['CC'] = 'gcc'

setup(name='lab',
      version='0.1.0',
      description='A generic interface for linear algebra backends',
      long_description=readme,
      author='Wessel Bruinsma',
      author_email='wessel.p.bruinsma@gmail.com',
      url='https://github.com/wesselb/lab',
      license=licence,
      packages=find_packages(exclude=('tests', 'docs')),
      cmdclass={'build_ext': build_ext},
      ext_modules=[Extension('lab.bvn_cdf',
                             sources=['lab/bvn_cdf/bvn_cdf.pyx'],
                             include_dirs=[np.get_include()],
                             extra_compile_args=['-fPIC',
                                                 '-O2',
                                                 '-fopenmp'],
                             extra_link_args=['lab/bvn_cdf/tvpack.o',
                                              '-lgfortran',
                                              '-L/usr/local/gfortran/lib',
                                              '-fopenmp'])])
