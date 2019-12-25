#!/bin/bash

DEPDIR=_deps
[ -e $DEPDIR ] || mkdir $DEPDIR
[ -e $DEPDIR/bin ] || mkdir $DEPDIR/bin
[ -e $DEPDIR/lib ] || mkdir $DEPDIR/lib
[ -e $DEPDIR/include ] || mkdir $DEPDIR/include
ADEPDIR=$PWD/$DEPDIR
CFLAGS+=" -fopenmp" CPPFLAGS+=" -I${ADEPDIR}/include -fopenmp" LDFLAGS+=" -L${ADEPDIR}/lib" ./configure --prefix=${ADEPDIR} --with-pic $@
make clean
make
make install
