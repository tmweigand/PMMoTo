Sphere Pack: Euclidean Distance Transform and Pore Size Distribution
==================================

This example demonstrates how to compute a **Euclidian distance transform** and a **pore size distribution (PSD)** of a packed bed of spheres using PMMoTo. 

To run this example with 8 MPI processes:

.. code-block:: bash

   mpirun -np 8 python examples/sphere_pack_psd/sphere_pack_psd.py


Step 1: Import Modules
----------------------

.. code-block:: python

   from mpi4py import MPI
   import pmmoto

Step 2: Load Sphere Pack Geometry
---------------------------------

The sphere pack geometry file defines the simulation domain extents and the positions and radii of individual spheres. The format of the file is::

      domain_x_min domain_x_max
      domain_y_min domain_y_max
      domain_z_min domain_z_max
      sphere_1_x sphere_1_y sphere_1_z sphere_1_radius
      sphere_2_x sphere_2_y sphere_2_z sphere_2_radius
      sphere_3_x sphere_3_y sphere_3_z sphere_3_radius
      ...


- The first three lines define the domain box for each axis
- Each subsequent line defines a sphere with:

   - Center coordinates: `x`, `y`, `z`
   - Radius: `r`

*Example:*

.. code-block:: text

   0.0 40.0
   0.0 40.0
   0.0 40.0
   10.0 10.0 10.0 4.5
   25.0 20.0 18.0 5.0
   30.0 30.0 10.0 3.0

In PMMoto, the file is read with:

.. code-block:: python

   sphere_pack_file = "examples/sphere_pack_psd/sphere_pack.in"
   spheres, domain_box = pmmoto.io.data_read.read_sphere_pack_xyzr_domain(
       sphere_pack_file
   )


Step 3: Initialize Simulation Domain
------------------------------------

Initialize the simulation domain by specifying the number of voxels, domain size from the sphere pack input file, number of subdomains, boundary types, and the rank for each MPI process. 

.. code-block:: python

   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()

   voxels = (401, 401, 401)
   subdomains = (2, 2, 2)
   boundary = pmmoto.BoundaryType.PERIODIC
   boundary_types = (
      (boundary, boundary),
      (boundary, boundary),
      (boundary, boundary),
   )

   sd = pmmoto.initialize(
       voxels=voxels,
       rank=rank,
       subdomains=subdomains,
       box=domain_box,
       boundary_types=boundary_types,
   )

The domain, which consists of eight subdomains, is represented below. By default, each subdomain shares 1 voxel with its neighbors. This value can be modified by specifying ``pad``. 

.. image:: /_static/examples/sphere_pack_psd/subdomains.png
   :alt: Domain
   :class: only-light
   :align: center
   :width: 60%



Step 4: Generate the Porous Media
------------------------------------

First, we will treat the spheres as solids. 

.. code-block:: python

   pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres, invert=False)

The pore space is shown below where the subdomain with `rank = 5` has been omitted:

.. image:: /_static/examples/sphere_pack_psd/pore_space.png
   :alt: Pore Space
   :class: only-light
   :align: center
   :width: 60%

Step 5: Perform a Euclidean Distance Transform
------------------------------------

The Euclidean distance transform provides the distance to the nearest solid for every pore voxel. The distance3 transform can be calculated and attached to a PMMoto ``porousmedia`` object by calling ``pm.distance`` which avoids recalculing the transformn. 

.. code-block:: python

   dist = pmmoto.filters.distance.edt(pm.img,sd)
   dist = pm.distance

The distance transform of the sphere pack is shown below. 

.. image:: /_static/examples/sphere_pack_psd/distance.png
   :alt: Domain
   :class: only-light
   :align: center
   :width: 60%


Step 6: Determine the Pore Size Distribution
------------------------------------

The pore size distribution of a porous media represents the largest size sphere that full fits at a given pore voxel. With PMMoto, the number of radii can be specified as ``num_radii``. Additioannly, by setting ``inltet=True``, the pore size distribution of only inlet connected voxels can be determined. 

.. code-block:: python

   psd = pmmoto.filters.porosimetry.pore_size_distribution(
         sd, pm, num_radii=25, inlet=False
   )

The pore size distribution of the sphere pack is shown below. 

.. image:: /_static/examples/sphere_pack_psd/psd.png
   :alt: Domain
   :class: only-light
   :align: center
   :width: 60%


Step 7: Generate PSD Plot
-------------------------

Generate a histogram of pore sizes as either a ``pdf`` or ``cdf``.

.. code-block:: python

   pmmoto.filters.porosimetry.plot_pore_size_distribution(
       "examples/sphere_pack_psd/pm", sd, psd, plot_type="pdf"
   )

.. image:: /_static/examples/sphere_pack_psd/pm_pore_size_distribution.png
   :alt: Domain
   :class: only-light
   :align: center
   :width: 60%

Step 8: Analysis on Inverted Pore Space
-------------------------------------------

As a check, invert the porous media so that the spheres represent the pore space and perform a Euclidean distrance transform and a pore size distribution analysis. 

.. code-block:: python

   invert_pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres, invert=True)
   invert_psd = pmmoto.filters.porosimetry.pore_size_distribution(
        sd, invert_pm, num_radii=25, inlet=False
   )

   invert_distance = invert_pm.distance

   pmmoto.filters.porosimetry.plot_pore_size_distribution(
       "examples/sphere_pack_psd/inverted_pm", sd, invert_psd, num_radii=25, inlet=False
   )

The inverted pore space is:

.. image:: /_static/examples/sphere_pack_psd/inverted_pore_space.png
   :alt: Inverted Pore Space
   :class: only-light
   :align: center
   :width: 60%


The sphere pack consists of spheres with a uniform radius of 1.0. The Euclidean distance transforms detemines the distance to the nearest solid voxel. 

.. image:: /_static/examples/sphere_pack_psd/invert_distance.png
   :alt: Inverted Pore Space Distance
   :class: only-light
   :align: center
   :width: 60%

The pore size distribution correctly estimates that maximum radius of 1.0 with some numerical artifacts that may be resolved with improved resolution. 

.. image:: /_static/examples/sphere_pack_psd/invert_psd.png
   :alt: Inverted Pore Space PSD
   :class: only-light
   :align: center
   :width: 60%

The probability distribution function correctly determines the uniform sphere size. 

.. image:: /_static/examples/sphere_pack_psd/inverted_pm_pore_size_distribution.png
   :alt: Domain
   :class: only-light
   :align: center
   :width: 60%

Step 9: Save Images
---------------------------

.. code-block:: python

   pmmoto.io.output.save_img(
       file_name="examples/sphere_pack_psd/image",
       subdomain=sd,
       img=pm.img,
       additional_img={
           "psd": psd,
           "dist": dist,
           "invert_pm": invert_pm.img,
           "invert_dist": invert_distance,
           "invert_psd": invert_psd,
       },
   )

Output
------

The expected output from a successful run is:

- :code:`image.pvti` and a folder :code:`image_proc` with eight :code:`.vti` files which can be opened in **Paraview**
- :code: Two `.png` files of a pdf of the pore size distribution. 

The code used to generate the plots in this example is located at :code:`examples/sphere_pack_psd/plot_sphere_pack_psd.py` and must be run with :code:`pvpython`, ParaView's Python interpreter.

