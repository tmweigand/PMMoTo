.. image:: _static/logo-1-text.png
   :alt: PMMoTo Logo
   :align: center
   :width: 750px

PMMoTo
====================

The **Porous Media Morphology and Topology (PMMoTo)** toolkit is an open-source Python library for analyzing, modeling, and characterizing the structure of porous materials. Built with Cython and C++ and designed for distributed memory systems using MPI, PMMoTo includes parallelized implementations of connected components analysis, morphological operations (e.g., dilation and erosion), and Euclidean distance transforms. Unlike many tools that focus on image-based analysis, PMMoTo is specifically designed for synthetically generated porous media, such as sphere packings and molecular dynamics simulations.


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   examples/index
   references


.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/modules
