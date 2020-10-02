#!/bin/bash

# Install healpy and nose

pip install nose healpy scipy parameterize

#### Install GSL2.0+ ####

wget ftp://ftp.gnu.org/gnu/gsl/gsl-2.5.tar.gz --passive-ftp && tar xzf gsl-2.5.tar.gz && cd gsl-2.5 &&  ./configure --enable-shared && make && sudo make install && cd ..
