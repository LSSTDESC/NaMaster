# Installing NaMaster at NERSC

The following steps should be followed to install NaMaster at NERSC:

## 1 Install libsharp
libsharp is a C library for spherical harmonic transforms. Follow these steps to install it before moving on to NaMaster:
1. Download libsharp from [its github repository](https://github.com/dagss/libsharp) and unzip the file.
2. From the libsharp folder run `autoreconf -i`, which will generate a `configure` file.
3. Run `./configure --enable-pic` and `make`
4. Create three directories: `bin`, `lib` and `include` in your home directory (unless they're already there). E.g. `mkdir $HOME/bin` etc.
5. Move the contents of `auto/bin`, `auto/lib` and `auto/include` to the corresponding folders you just created in your home directory.

Note that you should also have HEALPix installed. At NERSC, this is installed in 
`/global/common/cori/contrib/hpcosmo/hpcports_gnu-4.0/healpix-3.30.1_62c0405b-4.0/`
so just make sure this is included in your paths.

It's also useful to load anaconda 2.7 (e.g. `module load python/2.7-anaconda`).


## 2 Edit your .bashrc.ext
From your home directory, open the file `.bashrc.ext` with your favourite text editor and add the following lines at the end of it:
```
export PATH=$HOME/bin:$PATH
export LD_LIBRARY_PATH=$HOME/lib:/usr/common/software/gsl/2.1/intel/lib:/usr/common/software/cfitsio/3.370-reentrant/hsw/intel/lib:$LD_LIBRARY_PATH
export LDFLAGS="-L/usr/common/software/gsl/2.1/intel/lib -L/usr/common/software/cfitsio/3.370-reentrant/hsw/intel/lib -L$HOME/lib"
export CPPFLAGS="-I/usr/common/software/gsl/2.1/intel/include -I/usr/common/software/cfitsio/3.370-reentrant/hsw/intel/include -I$HOME/include"
export CC=cc
```
Note that, if you have previously editted either `$LDFLAGS` or `$CPPFLAGS`, the changes above should be appended to their already existing values to avoid messing up your environment.

## 3 Install NaMaster
Once libsharp has been installed, download or clone NaMaster from its [github repository](https://github.com/damonge/NaMaster) and follow these steps:
1. Before anything else, run the following commands to avoid static linking of the final NaMaster library. Doing so just once would be enough (i.e. no need to include this in your .bashrc.ext):
```
export CRAYPE_LINK_TYPE=dynamic
export XTPE_LINK_TYPE=dynamic
```
2. Run `./configure --prefix=$HOME`, `make` and `make install`.
3. Open the file `setup.py` with your favourite text editor and make sure that the variable `use_icc` is set to `True`.
4. Run `python setup.py install --user`. This will install the python module, `pymaster`.
5. To check that the installation worked, go to the `test` directory and run `python check.py`. If you see a bunch of small numbers and plots coming up after a while (and no errors occurred), you can congratulate yourself: you have a working version of NaMaster on NERSC!
