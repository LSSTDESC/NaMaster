#!/bin/bash

# Install chealpix

wget https://sourceforge.net/projects/healpix/files/Healpix_3.11/autotools_packages/chealpix-3.11.4.tar.gz && tar xzf chealpix-3.11.4.tar.gz && cd chealpix-3.11.4 && ./configure --enable-shared && make && sudo make install && cd ..

# Install healpy and nose

pip install nose healpy scipy

#### Install libsharp ####

git clone https://github.com/Libsharp/libsharp.git
cd libsharp
autoconf -i
./configure --enable-pic
make
mv auto/bin $TRAVIS_BUILD_DIR
mv auto/lib $TRAVIS_BUILD_DIR
mv auto/include $TRAVIS_BUILD_DIR
cd ..

#### Install GSL2.0+ ####

wget ftp://ftp.gnu.org/gnu/gsl/gsl-2.5.tar.gz && tar xzf gsl-2.5.tar.gz && cd gsl-2.5 &&  ./configure --enable-shared && make && sudo make install && cd ..

