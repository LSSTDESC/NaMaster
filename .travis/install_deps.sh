#!/bin/bash

# Install chealpix

wget https://sourceforge.net/projects/healpix/files/Healpix_3.11/autotools_packages/chealpix-3.11.4.tar.gz && tar xzf chealpix-3.11.4.tar.gz && cd chealpix-3.11.4 && ./configure --enable-shared && make && sudo make install && cd ..

# Install healpy and nose

pip install nose healpy scipy

# Install cfitsio
wget heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio-3.47.tar.gz && tar xzf heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio-3.47.tar.gz && cd cfitsio-3.47 && ./configure && make && sudo make install && cd ..

#### Install GSL2.0+ ####

wget ftp://ftp.gnu.org/gnu/gsl/gsl-2.5.tar.gz --passive-ftp && tar xzf gsl-2.5.tar.gz && cd gsl-2.5 &&  ./configure --enable-shared && make && sudo make install && cd ..

