#!/usr/bin/env python

from distutils.core import *
from distutils import sysconfig
import os.path

# Get numpy include directory (works across versions)
import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


use_icc=False #Set to True if you compiled libsharp with icc
if use_icc :
    libs=['nmt','fftw3','fftw3_omp','sharp','fftpack','c_utils','chealpix','cfitsio','gsl','gslcblas','m','gomp','iomp5']
    extra=['-openmp',]
else :
    libs=['nmt','fftw3','fftw3_omp','sharp','fftpack','c_utils','chealpix','cfitsio','gsl','gslcblas','m','gomp']
    extra=['-O4', '-fopenmp',]


_nmtlib = Extension("_nmtlib",
                    ["pymaster/namaster_wrap.c"],
                    libraries = libs,
                    include_dirs = [numpy_include, "../src/"],
                    extra_compile_args=extra,
                    )

setup(name = "pymaster",
      description = "Library for pseudo-Cl computation",
      author = "David Alonso",
      version = "0.1",
      packages = ['pymaster'],
      ext_modules = [_nmtlib],
      )
