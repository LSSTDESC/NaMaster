#!/usr/bin/env python
import sys
from setuptools import setup, Extension
from distutils.errors import DistutilsError
from setuptools.command.build_py import build_py as _build
from setuptools.command.develop import develop as _develop
import subprocess as sp
import os, sys


# Get numpy include directory (works across versions)
import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

c_compile_args=''
if '--enable-fftw-pthreads' in sys.argv:
    sys.argv.pop(sys.argv.index('--enable-fftw-pthreads'))
    FFTW_LIBS = ['fftw3', 'fftw3_threads', 'pthread']
    c_compile_args+='--enable-fftw-pthreads '
else:
    FFTW_LIBS = ['fftw3', 'fftw3_omp']

if '--disable-openmp' in sys.argv:
    sys.argv.pop(sys.argv.index('--disable-openmp'))
    USE_OPENMP = False
    c_compile_args+='--disable-openmp '
else:
    USE_OPENMP = True

libs = ['sharp2', 'cfitsio', 'gsl', 'gslcblas', 'm'] + FFTW_LIBS

use_icc = False  # Set to True if you compiled libsharp with icc
if use_icc:
    extra = []
    if USE_OPENMP:
        libs += ['gomp', 'iomp5']
    extra += ['-openmp']
else:
    extra = ['-O4']
    if USE_OPENMP:
        libs += ['gomp']
    extra += ['-fopenmp']

def _compile_libsharp():
    if not os.path.exists('_deps/include/libsharp2/sharp.h'):
        try:
            sp.check_call('./scripts/install_libsharp.sh',
                          shell=True)
        except:
            raise DistutilsError('Failed to install libsharp.')

def _compile_libchealpix():
    if not os.path.exists('_deps/lib/libchealpix.a'):
        try:
            sp.check_call('./scripts/install_libchealpix.sh',
                          shell=True)
        except:
            raise DistutilsError('Failed to install libchealpix.')

def _compile_libnmt():
    if not os.path.exists('_deps/lib/libnmt.a'):
        try:
            sp.check_call('./scripts/install_libnmt.sh ' +
                          c_compile_args, shell=True)
        except:
            raise DistutilsError('Failed to compile C library.')

class build(_build):
    """Specialized Python source builder."""
    def run(self):
        _compile_libsharp()
        _compile_libchealpix()
        _compile_libnmt()
        _build.run(self)

class develop(_develop):
    """Specialized Python develop mode."""
    def run(self):
        _compile_libsharp()
        _compile_libchealpix()
        _compile_libnmt()
        _develop.run(self)

_nmtlib = Extension("_nmtlib",
                    ["pymaster/namaster_wrap.c"],
                    extra_objects=["./_deps/lib/libnmt.a", "./_deps/lib/libchealpix.a"],
                    libraries=libs,
                    library_dirs=["./_deps/lib/"],
                    include_dirs=[numpy_include, "./src/","./_deps/include/"],
                    extra_compile_args=extra,
                    extra_link_args=extra
                    )

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="pymaster",
      version="1.5",
      author="David Alonso",
      author_email="david.alonso@physics.ox.ac.uk",
      description="Library for pseudo-Cl computation",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/LSSTDESC/NaMaster",
      cmdclass={'build_py': build, 'develop': develop},
      classifiers=[
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Operating System :: Unix',
          'Operating System :: MacOS'],
      packages=['pymaster'],
      ext_modules=[_nmtlib],
      )
