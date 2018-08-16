#!/bin/bash
pip install nose healpy
git clone https://github.com/Libsharp/libsharp.git
cd libsharp
autoconf
./configure
make
mv lib $TRAVIS_BUILD_DIR/
mv include $TRAVIS_BUILD_DIR/
mv bin $TRAVIS_BUILD_DIR/
export PATH=$HOME/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRAVIS_BUILD_DIR/lib
export LDFLAGS="-L$TRAVIS_BUILD_DIR/lib"
export CPPFLAGS="-I$TRAVIS_BUILD_DIR/include"
cd $HOME
git clone https://github.com/LSSTDESC/NaMaster.git
cd NaMaster
./configure --prefix $HOME=$TRAVIS_BUILD_DIR
make
make install
