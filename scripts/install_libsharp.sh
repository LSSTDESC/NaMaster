#!/bin/bash

#I stole this from pixell: https://github.com/simonsobs/pixell

DEPDIR=_deps
[ -e $DEPDIR ] || mkdir $DEPDIR
[ -e $DEPDIR/bin ] || mkdir $DEPDIR/bin
[ -e $DEPDIR/lib ] || mkdir $DEPDIR/lib
[ -e $DEPDIR/include ] || mkdir $DEPDIR/include
cd $DEPDIR
[ -e libsharp ] || git clone https://github.com/Libsharp/libsharp # do we want a frozen version?
cd libsharp
aclocal
if [ $? -eq 0 ]; then
    echo Found automake.
else
    echo ERROR: automake not found. Please install this or libsharp will not be installed correctly.
    exit 127
fi
autoconf
./configure --enable-pic
make
rm -rf python/
cp auto/bin/* ../bin/
cp auto/lib/* ../lib/
cp auto/include/* ../include/
