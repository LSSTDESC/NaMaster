#!/bin/bash

# Install chealpix

wget https://sourceforge.net/projects/healpix/files/Healpix_3.11/autotools_packages/chealpix-3.11.4.tar.gz && tar xzf chealpix-3.11.4.tar.gz && cd chealpix-3.11.4 && ./configure --enable-shared && make && sudo make install && cd ..

# Install healpy and nose

#pip install nose healpy

#### Install libsharp ####

git clone https://github.com/Libsharp/libsharp.git 
cd libsharp
autoconf -i
./configure --enable-pic
make
make install
mv auto/bin $TRAVIS_BUILD_DIR
mv auto/lib $TRAVIS_BUILD_DIR
mv auto/include $TRAVIS_BUILD_DIR
export PATH=$TRAVIS_BUILD_DIR/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRAVIS_BUILD_DIR/lib:/usr/local/lib
export LDFLAGS="-L$TRAVIS_BUILD_DIR/lib -L/usr/local/lib"
export CPPFLAGS="-I$TRAVIS_BUILD_DIR/include -I/usr/local/include -fopenmp"
export CFLAGS="-fopenmp"

#### Install GSL2.0+ ####

cd $HOME
wget http://mirror.rise.ph/gnu/gsl/gsl-2.4.tar.gz && tar xzf gsl-2.4.tar.gz && cd gsl-2.4 &&  ./configure --enable-shared && make && sudo make install && cd ..

#### Install NaMaster C ####

cd $HOME
git clone https://github.com/LSSTDESC/NaMaster.git
cd NaMaster
./configure --prefix=$TRAVIS_BUILD_DIR
make
sudo make install
make check
