Quickstart
==========

This example demonstrates how to generate a porous domain and save the result for visualization.

Single-Core Example
-------------------

The following script generates a random binary image using a single process (i.e., without parallel execution).

In parallel computing with MPI, a process is an independent instance of a program with its own memory space. Each process is assigned a unique integer rank used to distinguish it from other processes. In a single-core (or single-process) context, there is only one process with rank 0, and no parallelism is involved.

This example is useful for development, testing, or when running on systems without an MPI setup.

.. code-block:: python

   import pmmoto

   voxels = (100, 100, 100)
   sd = pmmoto.initialize(voxels)
   img = pmmoto.domain_generation.gen_img_random_binary(sd.voxels)

   pmmoto.io.output.save_img("output/image", sd, img)

This will create a file named ``output/image.vti``, which can be opened using **ParaView** for 3D visualization.

Parallel Example
----------------
Modifications to the workflow are necessary to enable execution in parallel using multiple processes. In this case, MPI launches separate processes, each assigned a unique **rank** from `0` to `N-1`, where `N` is the total number of processes. For this example, `N=2.`

PMMoTo uses the MPI ranks to **decompose the full domain into subdomains**, assigning one subdomain to each rank. For this reason, the number of MPI ranks must exactly match the number of subdomains specified in the script. All ranks work concurrently, with communication handled automatically through MPI.

This approach allows PMMoTo to scale efficiently across multiple cores or nodes in a distributed memory environment.


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

.. note::

   The maximum number of MPI ranks should not exceed the number of available CPU cores to avoid oversubscription and potential performance degradation. You can determine the number of available CPU cores in Python using:

   .. code-block:: python

      import os
      print(os.cpu_count())

   Alternatively, consult your machine's specifications or use system commands like ``nproc`` (Linux) or ``sysctl -n hw.ncpu`` (macOS) to find the core count. 
   
   However, the number of cores alone is not the full story as parameters like CPU architecture (performance vs. efficiency cores), memory bandwidth, and other factors also influence performance.

