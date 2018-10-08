#!/bin/bash

#### Install NaMaster C ####

if ![[ TRAVIS_OS_NAME == "linux" ]]; then
export CC=gcc-8
fi
export PATH=$TRAVIS_BUILD_DIR/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRAVIS_BUILD_DIR/lib:/usr/local/lib
export LDFLAGS="-L$TRAVIS_BUILD_DIR/lib -L/usr/local/lib"
export CPPFLAGS="-I$TRAVIS_BUILD_DIR/include -I/usr/local/include -fopenmp"
export CFLAGS="-fopenmp"

./configure --prefix=$TRAVIS_BUILD_DIR
make clean
make
sudo make install
make check
