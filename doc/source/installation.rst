************
Installation
************

There are different ways to install NaMaster. In rough order of complexity, they are:


Conda forge
===========

Unless you care about optimizing the code, it's worth giving this one a go. The conda recipe for NaMaster is currently hosted on `conda-forge <https://anaconda.org/conda-forge/namaster>`_ (infinite kudos to `Mat Becker <https://github.com/beckermr>`_ for this). In this case, installing NaMaster means simply running:

.. code-block:: bash
                
   $ conda install -c conda-forge namaster

If that works for you and you don't care about optimizing the code too much, skip the rest of this section. If you don't have admin permissions, you can give virtual environments a try (or else follow the instructions below).


PyPI
====

NaMaster is also hosted on `PyPI <https://pypi.org/project/pymaster>`_. Installing it should be as simple as running:

.. code-block:: bash

    $ python -m pip install pymaster [--user]

(add ``--user`` if you don't have admin permissions). Note that this will compile the code on your machine, so you'll need to have installed its :ref:`dependencies <dependencies>`.


From source
===========

If all the above fail, try to install NaMaster from its source. You should first clone this `github repository <https://github.com/LSSTDESC/NaMaster>`_. Then follow these steps:

1. Install dependencies.
------------------------

Install the dependencies listed :ref:`here <dependencies>`. Note that some of them (HEALPix) may not be necessary, as pymaster will attempt to install them automatically.

2. Install the python module
----------------------------

Installing the python module ``pymaster`` should be as simple as running

.. code-block:: bash

   $ python setup.py install [--user]

or, even better, if you can use ``pip``:

.. code-block:: bash

   $ pip install . [--user]

where the optional ``--user`` flag can be used if you don't have admin privileges.

You can check that the python installation works by running the unit tests:

.. code-block:: bash

   $ python -m unittest discover -v

Note that the ``test`` directory, containing all unit tests, also contains all the sample python scripts described in the rest of this documentation.

If you installed ``pymaster`` via ``pip``, you can uninstall everything by running

.. code-block:: bash

   $ pip uninstall pymaster


**Note that the C library is automatically compiled when installing the python module.** If you care about the C library at all, or you have trouble compiling it, see the next section.

3. Install the C code (optional)
--------------------------------

The script ``scripts/install_libnmt.sh`` contains the instructions run by ``setup.py`` to compile the C library (``libnmt.a``). You may have to edit this file or make sure to include any missing compilation flags if ``setup.py`` encounters issues compiling the library.

If you need the C library for your own code, ``scripts/install_libnmt.sh`` installs it in ``_deps/lib`` and ``_deps/include``. Note that the script process will also generate an executable ``namaster``, residing in ``_deps/bin`` that can be used to compute power spectra. The use of this program is discouraged over using the python module.

You can check that the C code works by running

.. code-block:: bash

   $ make check

If all the checks pass, you're good to go.


Installing on Mac
=================

NaMaster can be installed on Mac using any of the methods above as long as you have either the ``clang`` compiler with OpenMP capabilities or the ``gcc`` compiler. Both can be accessed via homebrew. If you don't have either, you can still try the conda installation above.

**Note: NaMaster is not supported on Windows machines yet.**


.. _dependencies:

Dependencies
============

NaMaster has the following dependencies, which should be present in your system before you can install the code from source:

* `GSL <https://www.gnu.org/software/gsl/>`_. Version 2 required.
* `FFTW <http://www.fftw.org/>`_. Version 3 required. Install with ``--enable-openmp`` and potentially also ``--enable-shared``.
* `cfitsio <https://heasarc.gsfc.nasa.gov/fitsio/>`_. Any version >3 should work.

Besides these, NaMaster will attempt to install the following additional dependency. If this fails, or if you'd like to use your own preinstalled versions, follow these instructions:

* `HEALPix <https://sourceforge.net/projects/healpix/>`_. HEALPix is automatically installed by ``setup.py`` by running the script ``scripts/install_libchealpix.sh`` (have a look there if you run into trouble). HEALPix gets installed in ``_deps/lib`` and ``_deps/include``. However, if you want to use your own preinstalled version , you should simlink it into the directory ``_deps``, such that ``_deps/lib/libchealpix.a`` can be seen. Any version >2 should work. Only the C libraries are needed.
