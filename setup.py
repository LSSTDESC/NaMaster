#!/usr/bin/env python
import sys
from setuptools import setup, Extension # RM
from distutils.errors import DistutilsError
from setuptools.command.build_py import build_py as _build # RM
from setuptools.command.develop import develop as _develop # RM
import subprocess as sp # RM
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

libs = [
    'sharp', 'fftpack', 'c_utils', 'cfitsio',
    'gsl', 'gslcblas', 'm'] + FFTW_LIBS

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

def _compile_libsharp(): # RM
    if not os.path.exists('_deps/lib/libsharp.a'): # RM
        try: # RM
            sp.check_call('./scripts/install_libsharp.sh', # RM
                          shell=True) # RM
        except: # RM
            raise DistutilsError('Failed to install libsharp.') # RM

def _compile_libchealpix(): # RM
    if not os.path.exists('_deps/lib/libchealpix.a'): # RM
        try: # RM
            sp.check_call('./scripts/install_libchealpix.sh', # RM
                          shell=True) # RM
        except: # RM
            raise DistutilsError('Failed to install libchealpix.') # RM

def _compile_libnmt(): # RM
    if not os.path.exists('_deps/lib/libnmt.a'): # RM
        try: # RM
            sp.check_call('./scripts/install_libnmt.sh ' + # RM
                          c_compile_args, shell=True) # RM
        except: # RM
            raise DistutilsError('Failed to compile C library.') # RM

class build(_build): # RM
    """Specialized Python source builder.""" # RM
    def run(self): # RM
        _compile_libsharp() # RM
        _compile_libchealpix() # RM
        _compile_libnmt() # RM
        _build.run(self) # RM

class develop(_develop): # RM
    """Specialized Python develop mode.""" # RM
    def run(self): # RM
        _compile_libsharp() # RM
        _compile_libchealpix() # RM
        _compile_libnmt() # RM
        _build.run(self) # RM

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
      version="1.1",
      author="David Alonso",
      author_email="david.alonso@physics.ox.ac.uk",
      description="Library for pseudo-Cl computation",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/LSSTDESC/NaMaster",
      cmdclass={'build_py': build, 'develop': develop}, # RM
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
