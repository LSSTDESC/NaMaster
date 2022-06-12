# Installing NaMaster at NERSC

The procedure below, making use of conda environments, should be followed to install NaMaster at NERSC (thanks to [Laurie Stephey](https://github.com/lastephey) from NERSC -- see [this issue](https://github.com/LSSTDESC/NaMaster/issues/154)):

```
#ssh to cori

module load python
module load cray-fftw
module unload craype-hugepages2M

conda create -n pymaster python=3.9 -y
conda activate pymaster
conda install gsl cfitsio numpy -c conda-forge

export CC=cc
export CRAYPE_LINK_TYPE=dynamic
export XTPE_LINK_TYPE=dynamic
export LDFLAGS+=" -L$CONDA_PREFIX/lib"
export CPPFLAGS+=" -I$CONDA_PREFIX/include"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
LDSHARED="cc -shared" CC=cc python -m pip install pymaster --force-reinstall --no-cache-dir

#test by

python -c "import pymaster" 
```
