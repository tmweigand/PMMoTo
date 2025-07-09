Quickstart
==========

This example demonstrates how to generate a porous domain and save the result to disk.

Single-Core Example
-------------------

The following script generates a random binary image on a single process:

.. code-block:: python

   import pmmoto

   voxels = (100, 100, 100)
   sd = pmmoto.initialize(voxels)
   img = pmmoto.domain_generation.gen_img_random_binary(sd.voxels)

   pmmoto.io.output.save_img("output/image", sd, img)

This will create a file named ``output/image.vti``, which can be opened using **ParaView** for 3D visualization.

Parallel Example
----------------

The same workflow can be run in parallel using multiple MPI ranks:

.. code-block:: python

   from mpi4py import MPI
   import pmmoto

   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()

   voxels = (100, 100, 100)
   subdomains = (2, 1, 1)  # Product must match total number of MPI ranks

   sd = pmmoto.initialize(voxels, rank=rank, subdomains=subdomains)
   img = pmmoto.domain_generation.gen_img_smoothed_random_binary(sd.voxels)

   pmmoto.io.output.save_img("parallel_output/image", sd, img)

Launch the script using:

.. code-block:: bash

   mpirun -np 2 python script_name.py

.. note::

   Parallel scripts must be launched from the command line using ``mpirun`` or ``mpiexec``.
   They will not run correctly in Jupyter notebooks or interactive Python shells unless
   special configurations (e.g., using ``ipymp``) are applied.

This will generate a `.pvti` file (e.g., ``parallel_output/image.pvti``), which can be opened in **ParaView** to visualize the full reconstructed image. Individual ranks also write `.vti` files corresponding to their subdomain.

