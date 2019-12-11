# Installing NaMaster at NERSC

The following steps should be followed to install NaMaster at NERSC (see also [this issue](https://github.com/LSSTDESC/NaMaster/issues/62) for additional instructions):

## 0 Python settings
First, pick your favourite python module at NERSC. NaMaster has been tested on both 2.7 and 3.6, so do either
```
module load python/2.7-anaconda-4.4
```
or
```
module load python/3.6-anaconda-5.2
```

You probably also want to have `healpy` installed:
```
pip install --user healpy
```

## 1 Prepare to compile the C libraries
You'll want to run the following in order to prepare your environment to install NaMaster:
```
module load gsl/2.5
export LDFLAGS+=" -L$GSL_DIR/lib -L$HOME/lib"
export CPPFLAGS+=" -I$GSL_DIR/include -I$HOME/include"
```
This makes sure that the OS will be able to find GSL and all the other libraries at compile time.

I'd also advice to add the following lines to your `.bashrc.ext` file (in your home directory), so python will always be able to find NaMaster.
```
module load gsl/2.5
export CC=cc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GSL_DIR/lib:$HOME/lib
```

Note that the above assumes you'll install all C dependencies below into `$HOME/include` and `$HOME/lib`. I find this a convenient way to keep things tidy at NERSC.

## 2 Install libsharp
libsharp is a C library for spherical harmonic transforms. Follow these steps to install it before moving on to NaMaster.
There are three different versions of libsharp you may want to install:
1. The github version from [its github repository](https://github.com/dagss/libsharp).
2. The gitlab version (latest) from [gitlab](https://gitlab.mpcdf.mpg.de/mtr/libsharp).
3. The version shipped with HEALPix.
We provide instructions for each of these here:

### 2.1 The github version (default)
1. Download libsharp from its github repository and unzip the file.
2. From the libsharp folder run `autoreconf -i`, which will generate a `configure` file.
3. Run `./configure --enable-pic` and `make`
4. Create three directories: `bin`, `lib` and `include` in your home directory (unless they're already there). E.g. `mkdir $HOME/bin` etc.
5. Move the contents of `auto/bin`, `auto/lib` and `auto/include` to the corresponding folders you just created in your home directory by hand.

### 2.2 The gitlab version
1. Download libsharp from its github repository and unzip the file.
2. From the libsharp folder run `autoreconf -i`, which will generate a `configure` file.
3. Follow the instruction on the `COMPILE` file to run `configure`. The command `CC=cc CFLAGS="-std=c99 -O3 -ffast-math" ./configure --prefix=$HOME` should work.
4. After running the `configure` script, type `make` and `make install`. To check that the installation worked run the test suite typing
```
export CRAYPE_LINK_TYPE=dynamic
export XTPE_LINK_TYPE=dynamic
make check
```
5. Finally, copy the header file in `pocketfft/pocketfft.h` to `$HOME/include` by hand, since NaMaster needs to use it.

### 2.3 The version shipped with HEALPix
Instructions provided in section 5 below.

## 3 Install cfitsio
1. Go to https://heasarc.gsfc.nasa.gov/fitsio/ and download the latest tarball. Untar it and go into its root directory.
2. Run `./configure CFLAGS=-fPIC --prefix=$HOME` (as I said above, I'm assuming that you'll install everything into your home directory).
3. Run `make; make install`.

## 4 Install FFTW
Doing `module load cray-fftw` should be enough to get FFTW to work with NaMaster. If you still get FFTW-related errors, you may have to install it from source (again, into your home directory preferably). To do so, download the latest tarball from http://www.fftw.org/, untar and do:
```
./configure --prefix=$HOME --with-openmp
make
make install
```

## 5 Install HEALPix
1. Download the latest version of HEALPix from its website https://healpix.jpl.nasa.gov/, untar and go into the root directory.
2. Run `./configure` and follow the instructions to install the C package (option 2). Make sure to mark `cc` as your preferred compiler. You don't need the shared library or the suggested modifications to your shell profile.
3. Run `make c-all`
4. Move (by hand!) the contents of `./lib/` and `./include/` to their corresponding equivalents in your `$HOME`.
5. If you want to use the libsharp version distributed with HEALPix with NaMaster you need to
download HEALPix version >=3.50 and configure the C++ package. For this purpose we suggest to use the GNU programming environment
```
module swap PrgEnv-intel PrgEnv-gnu
```
 Run `./configure` and follow the instructions to install the C++ package. We suggest to use the `optimized_gcc`option. Do not forget to add to your environment the HEALPix C++ related paths. If you used the optimized_gcc configuration these would be
```
export LDFLAGS+=" -L$HEALPIX/src/cxx/optimized_gcc/lib/"
export CPPFLAGS+=" -I$HEALPIX/src/cxx/optimized_gcc/include/"
export LD_LIBRARY_PATH+=${LD_LIBRARY_PATH}:$HEALPIX/src/cxx/optimized_gcc/lib/
```
or move the contents of these paths to their corresponding equivalents in your `$HOME`.

## 6 Install NaMaster
Once all the above has been installed, download or clone NaMaster from its [github repository](https://github.com/damonge/NaMaster) and follow these steps:
1. Before anything else, run the following commands to avoid static linking of the final NaMaster library. Doing so just once would be enough (i.e. no need to include this in your .bashrc.ext):
```
export CRAYPE_LINK_TYPE=dynamic
export XTPE_LINK_TYPE=dynamic
```
2. Run `./configure --prefix=$HOME`, `make` and `make install`. If you want to link NaMaster to the libsharp library distributed with HEALPix then run `./configure` adding the `--enable-libsharp_healpix` flag. If you want to link NaMaster to its gitlab version, add the `--enable-libsharp_gitlab` flag instead. If none of these are passed, it will be assumed that you have installed the github version of libsharp.
3. Run `LDSHARED="cc -shared" CC=cc python setup.py install --user` (note that the `LDSHARED` instruction is there to force setuptools to use the right linker at NERSC, but I've never needed it on other machines). This will install the python module, `pymaster`. As before, if you're linking with the healpix or gitlab versions of libsharp, then add `--enable-libsharp_healpix` or `--enable-libsharp_gitlab` to the `setup.py` command.
4. To check that the installation worked, go to the `test` directory and run `python check.py`. If you see a bunch of small numbers and plots coming up after a while (and no errors occurred), you can congratulate yourself: you have a working version of NaMaster on NERSC!
