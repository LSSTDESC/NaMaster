#!/bin/bash

DEPDIR=_deps
[ -e $DEPDIR ] || mkdir $DEPDIR
[ -e $DEPDIR/bin ] || mkdir $DEPDIR/bin
[ -e $DEPDIR/lib ] || mkdir $DEPDIR/lib
[ -e $DEPDIR/include ] || mkdir $DEPDIR/include
ADEPDIR=$PWD/$DEPDIR
CFLAGS="$CFLAGS -fopenmp" CPPFLAGS="$CPPFLAGS -I${ADEPDIR}/include -fopenmp" LDFLAGS="$LDFLAGS -L${ADEPDIR}/lib" ./configure --prefix=${ADEPDIR} --with-pic $@
make clean
make
make install
