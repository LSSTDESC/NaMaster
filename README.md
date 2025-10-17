# NaMaster
[![Build status](https://github.com/LSSTDESC/NaMaster/actions/workflows/ci.yml/badge.svg)](https://github.com/LSSTDESC/NaMaster/actions/workflows/ci.yml)
[![Docs Status](https://readthedocs.org/projects/namaster/badge/?version=latest)](http://namaster.readthedocs.io/)
[![Coverage Status](https://coveralls.io/repos/github/LSSTDESC/NaMaster/badge.svg?branch=master)](https://coveralls.io/github/LSSTDESC/NaMaster?branch=master)

NaMaster is a Python module to compute full-sky angular cross-power spectra of masked fields with arbitrary spin and an arbitrary number of known contaminants using a pseudo-Cl (aka MASTER) approach. The code also implements E/B-mode purification and is available in both full-sky and flat-sky modes, as well as supporting fields defined at the discrete positions of catalog sources.


## Installation

There are different ways to install NaMaster. In rough order of complexity, they are:

### Conda forge 
Unless you care about optimizing the code, it's worth giving this one a go. The conda recipe for NaMaster is currently hosted on [conda-forge](https://anaconda.org/conda-forge/namaster) (infinite kudos to [Mat Becker](https://github.com/beckermr) for this). In this case, installing NaMaster means simply running:
```
conda install -c conda-forge namaster
```
If that works for you and you don't care about optimizing the code too much, skip the rest of this section. If you don't have admin permissions, you can give virtual environments a try (or else follow the instructions below).

### PyPI
NaMaster is also hosted on [PyPI](https://pypi.org/project/pymaster). Installing it should be as simple as running:
```
python -m pip install pymaster [--user]
```
(add `--user` if you don't have admin permissions). Note that this will compile the code on your machine, so you'll need to have installed its [dependencies](#dependencies).

### From source
If all the above fail, try to install NaMaster from its source. You should first clone this [github repository](https://github.com/LSSTDESC/NaMaster). Then follow these steps:

#### 1. Install dependencies.
Install the dependencies listed [here](#dependencies). Note that some of them (HEALPix) may not be necessary, as pymaster will attempt to install them automatically.

#### 2. Install the python module
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
pytest -vv pymaster
```
Note that the `test` directory contains all the sample python scripts described in the [documentation](https://namaster.readthedocs.io). The tutorial notebooks are inside the `doc` directory. The unit tests themselves are in `pymaster/tests/`.

If you installed `pymaster` via `pip`, you can uninstall everything by running
```
pip uninstall pymaster
```

***Note that the C library underlying NaMaster is automatically compiled when installing the python module.*** If you care about the C library at all, or you have trouble compiling it, see the next section.

### 3. Install the C code (optional)
The script `scripts/install_libnmt.sh` contains the instructions run by `setup.py` to compile the C library (`libnmt.a`). You may have to edit this file or make sure to include any missing compilation flags if `setup.py` encounters issues compiling the library.

If you need the C library for your own code, `scripts/install_libnmt.sh` installs it in `_deps/lib` and `_deps/include`. Note that the script process will also generate an executable `namaster`, residing in `_deps/bin` that can be used to compute power spectra. The use of this program is discouraged over using the python module.

You can check that the C code works by running
```
make check
```
If all the checks pass, you're good to go.


## Installing on Mac

NaMaster can be installed on Mac using any of the methods above as long as you have either the `clang` compiler with OpenMP capabilities or the `gcc` compiler. Both can be accessed via homebrew. If you don't have either, you can still try the conda installation above.

***Note: NaMaster is not supported on Windows machines yet.***


## Documentation 
The following sources of documentation are available for users:
* [Python documentation](doc/build/html/index.html): also available in [readthedocs](http://namaster.readthedocs.io)
* [Scientific documentation](doc/doc_scientific.pdf): description of the methods implemented in NaMaster
* [C API documentation](doc/doc_C_API.pdf): description of the C library functionality and the NaMaster executable.


## Dependencies
NaMaster has the following dependencies, which should be present in your system before you can install the code from source:
* [GSL](https://www.gnu.org/software/gsl/). Version 2 required (note in certain systems you may also need to install `openblas` - see [this issue](https://github.com/LSSTDESC/NaMaster/issues/106).
* [FFTW](http://www.fftw.org/). Version 3 required. Install with `--enable-openmp` and potentially also `--enable-shared`.
* [cfitsio](https://heasarc.gsfc.nasa.gov/fitsio/). Any version >3 should work.

Besides these, NaMaster will attempt to install the following additional dependency. If this fails, or if you'd like to use your own preinstalled versions, follow these instructions:
* [HEALPix](https://sourceforge.net/projects/healpix/). HEALPix is automatically installed by `setup.py` by running the script `scripts/install_libchealpix.sh` (have a look there if you run into trouble). HEALPix gets installed in `_deps/lib` and `_deps/include`. However, if you want to use your own preinstalled version , you should simlink it into the directory `_deps`, such that `_deps/lib/libchealpix.a` can be seen. Any version >2 should work. Only the C libraries are needed.


## Licensing, credits and feedback
You are welcome to re-use the code, which is open source and freely available under terms consistent with BSD 3-Clause licensing (see [LICENSE](LICENSE)).

If you use NaMaster for any scientific publication, we kindly ask you to cite this github repository and the companion paper https://arxiv.org/abs/1809.09603. Special kudos should go to the following heroes for their contributions to the code:
- Mat Becker (@beckermr)
- Giulio Fabbian (@gfabbian)
- Martina Gerbino (@mgerbino)
- Daniel Lenz (@DanielLenz)
- Zack Li (@xzackli)
- Thibaut Louis (@thibautlouis)
- Tom Cornish (@tmcornish)
- Kevin Wolz (@kwolz)

For feedback, please contact the author via github issues or email (david.alonso@physics.ox.ac.uk).
