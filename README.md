# NaMaster

NaMaster is a C library, Python module and standalone program to compute full-sky angular cross-power spectra of masked, spin-0 and spin-2 fields with an arbitrary number of known contaminants using a pseudo-Cl (aka MASTER) approach. The code also implements E/B-mode purification and is available in both full-sky and flat-sky modes.


## Installation
NaMaster has the following dependencies, which should be present in your system before you can install the code:
* [GSL](https://www.gnu.org/software/gsl/). Version 2 required.
* [FFTW](http://www.fftw.org/). Version 3 required. Install with `--enable-openmp`.
* [libsharp](https://github.com/Libsharp/libsharp) (see instructions in [NERSC_installation.md](NERSC_installation.md) for more details on how to install libsharp).
* [cfitsio](https://heasarc.gsfc.nasa.gov/fitsio/). Any version >3 should work.
* [HEALPix](https://sourceforge.net/projects/healpix/). Any version >2 should work. You only need to install the C libraries.

### 1- C library
First, install the C library `libnmt`. In UNIX, in the simplest case, this should be a matter of running
```
./configure
make
make install
```
where the last command should be preceded by `sudo` if you need (and can get) admin privileges. If you don't have admin privileges, you can change the first command to
```
./configure --prefix=/path/to/install
```
where `/path/to/install` is an absolute path to the directory where the C library and include files will be installed.

If you have installed the C library in a non-standard path, you may have to add
```
export LD_LIBRARY_PATH=/path/to/install/lib:$LD_LIBRARY_PATH
```
to your `.bashrc` or `.bash_profile` for your system to be able to find `libnmt`.

Note that the installation process will also generate an executable `namaster`, residing in `/path/to/install/bin` that can be used to compute power spectra. The use of this program is discouraged over using the C library or python module.

Once you have installed the C library, you can check that everything works by running
```
make check
```
If all the checks pass, you're good to go.

### 2- Python module
Installing the python module `pymaster` should be as simple as running
```
python setup.py install [--user]
```
where the optional `--user` flag can be used if you don't have admin privileges.

You can check that the python installation works by running the unit tests:
```
python -m unittest discover -v
```
Note that the `test` directory, containing all unit tests, also contains all the sample python scripts described in the documentation (see below).


## Documentation 
The following sources of documentation are available for users:
* [Scientific documentation](doc/doc_scientific.pdf): description of the methods implemented in NaMaster
* [C API documentation](doc/doc_C_API.pdf): description of the C library functionality and the NaMaster executable. Installation instructions and a description of all dependencies can also be found here.
* [Python wrapper documentation](doc/build/html/index.html): also available in [readthedocs](http://namaster.readthedocs.io/en/latest/)


## Licensing, credits and feedback
You are welcome to re-use the code, which is open source and freely available under terms consistent with BSD 3-Clause licensing (see [LICENSE](LICENSE)).

If you use NaMaster for any scientific publication, we kindly ask you to cite this github repository and the companion paper https://arxiv.org/abs/1809.09603.

For feedback, please contact the author via github issues or emaild (david.alonso@physics.ox.ac.uk).
