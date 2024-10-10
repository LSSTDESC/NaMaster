.. pymaster documentation master file, created by
   sphinx-quickstart on Fri Feb 17 04:34:56 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NaMaster's documentation!
====================================

NaMaster is a python package that provides support to compute the angular power spectrum of masked fields with arbitrary spin pymaster using the so-called pseudo-Cl formalism. ``pymaster`` is the python implementation of the NaMaster library. Below you can find links to NaMaster's full documentation, a series of detailed tutorials in the form of ipython notebooks, and various example scripts showcasing its usage. Understanding these notebooks and scripts will allow you to make the most efficient use of this package.

We recommend that users read NaMaster's :download:`scientific documentation <doc_scientific.pdf>`, as well as the original paper `Alonso et al. 2019 <https://arxiv.org/abs/1809.09603>`_ to have a good understanding of the methods implemented in the library. Additional useful information, particularly regarding the calculation of covariance matrices, can be found in `Garcia-Garcia et al. 2019 <https://arxiv.org/abs/1906.11765>`_, and `Nicola et al. 2021 <https://arxiv.org/abs/2010.09717>`_. Support for catalog-based fields was documented in `Wolz et al. 2024 <https://arxiv.org/abs/2407.21013>`_, and the estimator's extension to anisotropic weighting of spin-s fields was presented in `Alonso 2024 <https://arxiv.org/abs/2410.07077>`_. We kindly request that you cite these papers where relevant.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   source/installation
   API Documentation<api/pymaster>
   source/tutorials
   source/sample_simple
   source/sample_bins
   source/sample_fields
   source/sample_masks
   source/sample_workspaces
   source/sample_pureb
   source/sample_flat
   source/sample_covariance
   source/sample_rectpix
   source/sample_toeplitz
   source/sample_shearcatalog
   source/sample_clusteringcatalog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
