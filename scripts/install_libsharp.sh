#!/bin/bash

#I stole this from pixell: https://github.com/simonsobs/pixell

DEPDIR=_deps
[ -e $DEPDIR ] || mkdir $DEPDIR
[ -e $DEPDIR/bin ] || mkdir $DEPDIR/bin
[ -e $DEPDIR/lib ] || mkdir $DEPDIR/lib
[ -e $DEPDIR/include ] || mkdir $DEPDIR/include
cd $DEPDIR
[ -e libsharp2 ] || git clone https://gitlab.mpcdf.mpg.de/mtr/libsharp.git libsharp2
cd libsharp2
aclocal
if [ $? -eq 0 ]; then
    echo Found automake.
else
    echo ERROR: automake not found. Please install this or libsharp will not be installed correctly.
    exit 127
fi
autoreconf -i
if [[ $TRAVIS ]] ; then
    echo "Compiling on travis"
    CFLAGS="-std=c99 -O3 -ffast-math"
else
    echo "Using -march=native. Binary will not be portable."
    CFLAGS="-march=native -std=c99 -O3 -ffast-math"
fi
CFLAGS=$CFLAGS ./configure --prefix=${PWD}/../ --enable-shared=no --with-pic=yes
make
make install
# Needed to create the CAR geomhelper
cp libsharp2/pocketfft.h ../include/libsharp2/
rm -rf python/
