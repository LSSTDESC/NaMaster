# Installing NaMaster at NERSC

The following steps should be followed to install NaMaster at NERSC (see also [this issue](https://github.com/LSSTDESC/NaMaster/issues/62) for additional instructions you may potentially need):

## Prepare your system
First, follow these steps to make sure the installation goes smoothly. We assume that you're using NERSC's Intel C compiler, and that you haven't loaded any modules that conflict with the ones below. It may be a good idea to add these commands to your `.bashrc.ext` file, but things should also work if you just type them before installing NaMaster.

1. Load the following modules, corresponding to the different dependencies of NaMaster and to anaconda's python
```
module load python/3.6-anaconda-5.2
module load gsl/2.5
module load cfitsio/3.47
module load cray-fftw
```

2. Make sure the C compiler can find these libraries. This involves adding the relevant paths to the `LDFLAGS`, `CPPFLAGS` and `LD_LIBRARY_FLAGS`:
```
export LDFLAGS+=" -L$GSL_DIR/lib -L$CFITSIO_DIR/lib"
export CPPFLAGS+=" -I$GSL_DIR/include -I$CFITSIO_DIR/include"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GSL_DIR/lib:$CFITSIO_DIR/lib
```

3. Select the preferred CC compiler (NERSC's `cc`)
```
export CC=cc
```

4. Run the following commands to avoid static linking of the final NaMaster library. Doing so just once should be enough (i.e. no need to include this in your `.bashrc.ext`):
```
export CRAYPE_LINK_TYPE=dynamic
export XTPE_LINK_TYPE=dynamic
```

## Install NaMaster
Run the following command to install NaMaster from pip:
```
LDSHARED="cc -shared" CC=cc python -m pip install pymaster --user
```
Note that the `LDSHARED` instruction is there to force setuptools to use the right linker at NERSC, but I've never needed it on other machines.

If you can't install the code from PyPI (e.g. if you want to install your own modified version), then go to the root directory of  NaMaster and run:
```
LDSHARED="cc -shared" CC=cc python setup.py install --user
```

## Running NaMaster on the compute nodes
Finally, there's one quirk of NERSC that should be taken into account (see [this issue](https://github.com/LSSTDESC/NaMaster/issues/62)). Before you run any script using NaMaster on NERSC (either on the queues or on an interactive node), run the following command:
```
module unload craype-hugepages2M
```

This has been an issue for a while, but it may be that by the time you install NaMaster, it has been sorted out, so it may be worth first trying to load pymaster on an interactive node, and then use this if it crashes.
