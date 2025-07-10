=============
Installation
=============

For best performance and support for distributed memory systems, PMMoTo requires a working MPI environment.

You may install **OpenMPI** or **MPICH**, depending on your system and preference:

- On macOS (with Homebrew):

  .. code-block:: bash

     brew install open-mpi
     # or
     brew install mpich

- On Ubuntu/Debian:

  .. code-block:: bash

     sudo apt install libopenmpi-dev openmpi-bin
     # or
     sudo apt install libmpich-dev mpich

Once MPI is installed, install PMMoTo from PyPI:

.. code-block:: bash

   pip install pmmoto

If you're installing from source for the first time:

.. code-block:: bash

   git clone https://github.com/tmweigand/pmmoto.git
   cd pmmoto
   pip install -e .

To visualize PMMoTo's `.vti` and `.vtp` output files, install `ParaView <https://www.paraview.org/>`_.

- On macOS:

  .. code-block:: bash

     brew install --cask paraview

- On Ubuntu/Debian:

  .. code-block:: bash

     sudo apt install paraview

- Or download the latest version directly from: https://www.paraview.org/download/
