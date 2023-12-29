#!/bin/bash

DEPDIR=_deps
[ -e $DEPDIR ] || mkdir $DEPDIR
[ -e $DEPDIR/bin ] || mkdir $DEPDIR/bin
[ -e $DEPDIR/lib ] || mkdir $DEPDIR/lib
[ -e $DEPDIR/include ] || mkdir $DEPDIR/include
autoreconf -ivf
ADEPDIR=$PWD/$DEPDIR
CFLAGS="$CFLAGS -fopenmp -O3 " CPPFLAGS="$CPPFLAGS -I${ADEPDIR}/include -fopenmp -O3" LDFLAGS="$LDFLAGS -L${ADEPDIR}/lib" ./configure --prefix=${ADEPDIR} --with-pic $@
if [ $? -eq 0 ]; then
    echo "Successful configure."
else
    echo "ERROR: failed to configure libnmt. Check all dependencies are installed"
    echo "       Dependencies:"
    echo "       - GSL"
    echo "       - FFTW"
    echo "       - CFITSIO"
    echo "       - HEALPix"
    exit 127
fi
make clean
make
if [ $? -eq 0 ]; then
    echo "Successful make."
else
    echo "ERROR: couldn't compile namaster. Make sure all dependencies are accessible."
    echo "       You may need to add the correct paths to CPPFLAGS, LDFLAGS and LD_LIBRARY_PATHS."
    echo "       E.g.:"
    echo "         >$ export CPPFLAGS+=\" -I/path/to/deps/include\""
    echo "         >$ export LDFLAGS+=\" -L/path/to/deps/lib\""
    echo "         >$ export LD_LIBRARY_PATHS=\$LD_LIBRARY_PATH:/path/to/deps/lib"
    echo " "
    echo "       Dependencies:"
    echo "       - GSL"
    echo "       - FFTW"
    echo "       - CFITSIO"
    echo "       - HEALPix"
    exit 127
fi
make install
