.. image:: _static/logo-1-text.png
   :alt: PMMoTo Logo
   :align: center
   :width: 750px

PMMoTo
====================

The Porous Media Morphology and Topology (PMMoTo) toolkit is an open-source Python library—backed by Cython and C++—for analyzing, modeling, and characterizing the structure of porous materials. The code is designed for performance on distributed memory systems using MPI. It includes fast connected components analysis, morphological operations (e.g., addition and subtraction), and Euclidean distance transforms. Output images are optimized for visualization in ParaView. PMMoTo also provides processing and analysis tools for reconstructing porous structures from molecular dynamics simulations.


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/drainage_inkbottle
   examples/sphere_pack_psd
   examples/connected_pathways
   examples/md_porous_media


.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/modules
