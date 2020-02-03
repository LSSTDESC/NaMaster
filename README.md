# NaMaster
[![Build Status](https://travis-ci.org/LSSTDESC/NaMaster.svg?branch=master)](https://travis-ci.org/LSSTDESC/NaMaster)
[![Docs Status](https://readthedocs.org/projects/namaster/badge/?version=latest)](http://namaster.readthedocs.io/)

NaMaster is a C library, Python module and standalone program to compute full-sky angular cross-power spectra of masked, spin-0 and spin-2 fields with an arbitrary number of known contaminants using a pseudo-Cl (aka MASTER) approach. The code also implements E/B-mode purification and is available in both full-sky and flat-sky modes.


## Installation
Unless you care a lot about optimizing the code, you should probably use the conda recipe for NaMaster currently hosted on [conda-forge](https://anaconda.org/conda-forge/namaster) (infinite kudos to [Mat Becker](https://github.com/beckermr) for this). This means simply running:
```
conda install -c conda-forge namaster
```
If that works for you and you don't care about optimizing the code too much, skip the rest of this section. If you don't have admin permissions, you can give virtual environments a try (or else follow the instructions below).

### 0- Dependencies
NaMaster has the following dependencies, which should be present in your system before you can install the code:
* [GSL](https://www.gnu.org/software/gsl/). Version 2 required.
* [FFTW](http://www.fftw.org/). Version 3 required. Install with `--enable-openmp` and potentially also `--enable-shared`.
* [libsharp](https://github.com/Libsharp/libsharp). Libsharp is automatically installed with NaMaster (see section 3 below). However, if you want to use your own preinstalled version, you should simlink it into the directory `_deps`, such that `_deps/lib/libsharp.a` can be seen (see instructions in [NERSC_installation.md](NERSC_installation.md) for more details on libsharp).
* [cfitsio](https://heasarc.gsfc.nasa.gov/fitsio/). Any version >3 should work.
* [HEALPix](https://sourceforge.net/projects/healpix/). Any version >2 should work. You only need to install the C libraries (including the shared ones).

### 1- Python
Installing the python module `pymaster` should be as simple as running
```
python setup.py install [--user]
```
or, even better, if you can use `pip`:
```
pip install . [--user]
```
where the optional `--user` flag can be used if you don't have admin privileges.

You can check that the python installation works by running the unit tests:
```
python -m unittest discover -v
```
Note that the `test` directory, containing all unit tests, also contains all the sample python scripts described in the documentation (see below).

If you installed `pymaster` via `pip`, you can uninstall everything by running
```
pip uninstall pymaster
```

***Note that the C library is automatically compiled when installing the python module.*** If you care about the C library at all, or you have trouble compiling it, see the next section.

### 2- C library
The script `scripts/install_libnmt.sh` contains the instructions run by `setup.py` to compile the C library (`libnmt.a`). You may have to edit this file or make sure to include any missing compilation flags if `setup.py` encounters issues compiling the library.

If you need the C library for your own code, `scripts/install_libnmt.sh` installs it in `_deps/lib` and `_deps/include`. Note that the script process will also generate an executable `namaster`, residing in `_deps/bin` that can be used to compute power spectra. The use of this program is discouraged over using the python module.

You can check that the C code works by running
```
make check
```
If all the checks pass, you're good to go.

### 3- Libsharp
`setup.py` attempts to download and install libsharp automatically. This is done by running the script `scripts/install_libsharp.sh`. If you encounter any trouble during this step, inspect the contents of that file. Libsharp gets installed in `_deps/lib` and `_deps/include`.


## Documentation 
The following sources of documentation are available for users:
* [Scientific documentation](doc/doc_scientific.pdf): description of the methods implemented in NaMaster
* [C API documentation](doc/doc_C_API.pdf): description of the C library functionality and the NaMaster executable. Installation instructions and a description of all dependencies can also be found here.
* [Python wrapper documentation](doc/build/html/index.html): also available in [readthedocs](http://namaster.readthedocs.io/en/latest/)


## Licensing, credits and feedback
You are welcome to re-use the code, which is open source and freely available under terms consistent with BSD 3-Clause licensing (see [LICENSE](LICENSE)).

If you use NaMaster for any scientific publication, we kindly ask you to cite this github repository and the companion paper https://arxiv.org/abs/1809.09603. Special kudos should go to the following heroes for their contributions to the code:
- Mat Becker (@beckermr)
- Giulio Fabbian (@gfabbian)
- Daniel Lenz (@DanielLenz)
- Zack Li (@xzackli)
- Thibaut Louis (@thibautlouis)

For feedback, please contact the author via github issues or emaild (david.alonso@physics.ox.ac.uk).
