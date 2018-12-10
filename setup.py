#!/usr/bin/env python
import sys
from setuptools import setup, Extension

# Get numpy include directory (works across versions)
import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

if '--enable-fftw3-pthreads' in sys.argv:
    sys.argv.pop(sys.argv.index('--enable-fftw3-pthreads'))
    FFTW_LIBS = ['fftw3', 'fftw3_threads', 'pthread']
else:
    FFTW_LIBS = ['fftw3', 'fftw3_omp']


use_icc=False #Set to True if you compiled libsharp with icc
if use_icc :
    libs=['nmt','sharp','fftpack','c_utils','chealpix','cfitsio','gsl','gslcblas','m','gomp','iomp5'] + FFTW_LIBS
    extra=['-openmp',]
else :
    libs=['nmt','fftw3','fftw3_omp','sharp','fftpack','c_utils','chealpix','cfitsio','gsl','gslcblas','m','gomp'] + FFTW_LIBS
    extra=['-O4','-fopenmp']


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
