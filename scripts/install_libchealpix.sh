#!/bin/bash

#Javi Sanchez is the dude

DEPDIR=_deps
[ -e $DEPDIR ] || mkdir $DEPDIR
[ -e $DEPDIR/bin ] || mkdir $DEPDIR/bin
[ -e $DEPDIR/lib ] || mkdir $DEPDIR/lib
[ -e $DEPDIR/include ] || mkdir $DEPDIR/include
ADEPDIR=$PWD/$DEPDIR
cd $DEPDIR
unameOut="$(uname -s)"
echo ${unameOut}
if [ "${unameOut}" = "Linux" ]
then
    [ -e chealpix-3.11.4 ] || wget https://sourceforge.net/projects/healpix/files/Healpix_3.11/autotools_packages/chealpix-3.11.4.tar.gz && tar xzf chealpix-3.11.4.tar.gz
elif [ "${unameOut}" = "Darwin" ]
then
    [ -e chealpix-3.11.4 ] || curl https://sourceforge.net/projects/healpix/files/Healpix_3.11/autotools_packages/chealpix-3.11.4.tar.gz -L --output chealpix-3.11.4.tar.gz && tar xzf chealpix-3.11.4.tar.gz
fi
cd chealpix-3.11.4
./configure --enable-static --disable-shared --with-pic --prefix=${ADEPDIR} $@
if [ $? -eq 0 ]; then
    echo "Successful configure."
else
    echo "ERROR: failed to configure HEALPix. Check CFITSIO is installed and reachable."
    exit 127
fi
make
if [ $? -eq 0 ]; then
    echo "Successful make."
else
    echo "ERROR: couldn't compile HEALPix. Make sure CFITSIO is installed."
    echo "       You may need to add the correct path to CPPFLAGS, LDFLAGS and LD_LIBRARY_PATHS."
    echo "       E.g.:"
    echo "         >$ export CPPFLAGS+=\" -I/path/to/cfitsio/include\""
    echo "         >$ export LDFLAGS+=\" -L/path/to/cfitsio/lib\""
    echo "         >$ export LD_LIBRARY_PATHS=\$LD_LIBRARY_PATH:/path/to/cfitsio/lib"
    exit 127
fi
make install
