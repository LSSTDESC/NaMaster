#!/bin/bash

# If we are using OSX, then install fftw3 and gsl2

if ! [[ $TRAVIS_OS_NAME == "linux" ]]; brew install gcc7; export CC=gcc-7; brew install fftw --with-openmp --without-fortran; brew install autoconf; 

    # Install some custom requirements on OS X
    if test -e $HOME/miniconda/bin; then
      echo "miniconda already installed.";
    else
      echo "Installing miniconda.";
      rm -rf $HOME/miniconda;
      mkdir -p $HOME/download;
      if [ "${TOXENV}" = py27 ]; then
        wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh -O $HOME/download/miniconda.sh;
      else
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O $HOME/download/miniconda.sh;
      fi;
      bash $HOME/download/miniconda.sh -b -p $HOME/miniconda;
    fi;
    source $HOME/miniconda/bin/activate
    hash -r
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
    conda info -a

    case "${TOXENV}" in
        py27)
            conda create -q -n test-environment python=2.7 pip
            ;;
        py36)
            conda create -q -n test-environment python=3.6 pip
            ;;
    esac;
fi;

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

wget ftp://ftp.gnu.org/gnu/gsl/gsl-2.5.tar.gz --passive-ftp && tar xzf gsl-2.5.tar.gz && cd gsl-2.5 &&  ./configure --enable-shared && make && sudo make install && cd ..

