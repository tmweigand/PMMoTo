=============
Installation
=============

PMMoTo requires a functioning MPI environment to achieve optimal performance on distributed memory systems.
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

.. note::
   For testing of PMMoTo, the developer mode ``dev`` must be specified to install the necessary dependencies. Please see :doc:`../developer_guide/installation` for detailed instructions. 



To visualize PMMoTo's `.vti` and `.vtp` output files, install `ParaView <https://www.paraview.org/>`_.

- On macOS:

  .. code-block:: bash

     brew install --cask paraview

- On Ubuntu/Debian:

  .. code-block:: bash

     sudo apt install paraview

.. note::
   For some systems, the latest executable version of ParaView may be required and can be found at: https://www.paraview.org/download/ 
