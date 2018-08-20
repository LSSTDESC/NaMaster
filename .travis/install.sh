#!/bin/bash

#### Install NaMaster C ####

./configure --prefix=$TRAVIS_BUILD_DIR
make clean
make
sudo make install
make check
