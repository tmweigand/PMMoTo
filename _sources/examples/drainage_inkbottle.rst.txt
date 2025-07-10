Drainage of an Ink Bottle
=========================

This example demonstrates a morphological drainage simulation in an ink bottle using PMMoTo. Two drainage approaches are compared:

- A standard morphological model :cite:`Hilpert_Miller_2001`
- A contact angle-based model :cite:`Schulz_Becker_2007`

To run this example:

.. code-block:: bash

   mpirun -np 2 python examples/drainage_inkbottle/drainage_inkbottle.py

Step 1: Import Modules
----------------------

.. code-block:: python

   from mpi4py import MPI
   import numpy as np
   import matplotlib.pyplot as plt
   import pmmoto

This script uses MPI for parallelism, `numpy` and `matplotlib` for postprocessing, and `pmmoto` for domain generation and simulation.

Step 2: Set Up Simulation Domain
--------------------------------

.. code-block:: python

    # Domain voxel resolution.
    # Must be a 3-tuple as only 3D is currently supported.
    voxels = (560, 120, 120)

    # Number of voxels allocated for the inlet reservoir region.
    reservoir_voxels = 20

    # Domain decomposition across MPI ranks.
    # The product of subdomain counts must match the number of MPI processes.
    subdomains = (2, 1, 1)

    # Physical extent of the domain in each dimension: (min, max)
    box = (
        (0.0, 14.0),  # x-dimension
        (-1.5, 1.5),  # y-dimension
        (-1.5, 1.5),  # z-dimension
    )

    # Boundary conditions for each face (−, +) per axis.
    # Options:
    #   - END:     Nothing assumed
    #   - WALL:    Solid, impermeable boundary added to image
    #   - PERIODIC: Wraparound — both faces of the axis must be periodic
    boundary_types = (
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),  # x
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),  # y
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),  # z
    )

    # Inlet boundary condition: (−, +) per axis
    # Used to specify where fluid enters the domain
    inlet = (
        (False, True),  # x: fluid enters from the +x face
        (False, False),  # y: no inlet
        (False, False),  # z: no inlet
    )

    # Outlet boundary condition: (−, +) per axis
    # Used to specify where fluid exits the domain
    outlet = (
        (True, False),  # x: fluid exits from the −x face
        (False, False),  # y: no outlet
        (False, False),  # z: no outlet
    )



Initialize the simulation domain with number of voxels and MPI parameters, specifying the decomposition (subdomains), boundary conditions and inlet/outlet, reservoir size, and global domain size for this MPI rank.

.. code-block:: python

   sd = pmmoto.initialize(
        voxels=voxels,
        box=box,
        boundary_types=boundary_types,
        rank=rank,
        subdomains=subdomains,
        inlet=inlet,
        outlet=outlet,
        reservoir_voxels=reservoir_voxels,
   )

The domain, which consists of two subdomains, is represented below:

.. image:: /_static/examples/drainage_inkbottle/subdomains.png
   :alt: Domain
   :class: only-light
   :align: center
   :width: 60%

Step 3: Generate Ink Bottle Geometry
------------------------------------

To create a traditional ink bottle as described in :cite:`Miller_Bruning_19` and given as 

.. math::
   y=0.01\cos(0.01x) + 0.5sin(x) + 0.75 \quad \forall x \in [0,14]

the ``domain_generation`` module in PMMoTo is used to provide a ``porous media`` object. 

.. code-block:: python

   pm = pmmoto.domain_generation.gen_pm_inkbottle(sd)


The pore space and reservoir of the ink bottle is shown below:

.. image:: /_static/examples/drainage_inkbottle/ink_bottle.png
   :alt: Ink bottle pore geometry
   :class: only-light
   :align: center
   :width: 60%

Step 4: Initialize Multiphase System
------------------------------------

Initialize a ``multiphase`` system and fill the pore space the wetting phase (fluid ID = 2).

.. code-block:: python

   mp = pmmoto.domain_generation.gen_mp_constant(pm, 2)

Step 5: Define Capillary Pressure Range
---------------------------------------

Create a sequence of capillary pressures designed to resolve a range of pore throat sizes for the ink bottle geometry. 

.. code-block:: python

   capillary_pressure = 0.1 + np.linspace(0, 1, 41) ** 1.5 * 7.6



Step 6: Perform Standard Morphological Drainage
-----------------------------------------------

Simulates drainage using the standard approach with a surface tension (gamma) of 1 :math:`\mathrm{mass}/\mathrm{seconds}^2`. The output of this function is the predicted equilibrium saturation at a given capillary pressure.

.. code-block:: python

   w_saturation_standard = pmmoto.filters.equilibrium_distribution.drainage(
       mp, capillary_pressure, gamma=1, method="standard"
   )


Step 7: Save Images
-------------------

Save the porous media image and the multiphase image at the last capillary pressure. The multiphase image ``mp.img`` is overwritten at every capillary pressure. Switching :code:`save=True` saves every multiphase image. 

.. code-block:: python

   pmmoto.io.output.save_img(
       file_name="examples/drainage_inkbottle/image",
       subdomain=sd,
       img=pm.img,
       additional_img={"mp_img": mp.img},
   )

A cross-section of the multiphase image is shown below:

.. image:: /_static/examples/drainage_inkbottle/standard_drainage.png
   :alt: Multiphase image
   :class: only-light
   :align: center
   :width: 60%


Step 8: Drainage with Contact Angle
-----------------------------------

Refill the pore space with the wetting fluid (fluid id = 2) and run the contact angle model where we set the contact angle to 20° and keep the surface tension at 1 :math:`\mathrm{mass}/\mathrm{seconds}^2`.

.. code-block:: python

   mp = pmmoto.domain_generation.gen_mp_constant(pm, 2)
   w_saturation_contact_angle = pmmoto.filters.equilibrium_distribution.drainage(
       mp, capillary_pressure, gamma=1, contact_angle=20, method="contact_angle"
   )

Step 9: Plot Results
--------------------

Generate a capillary pressure vs. saturation plot to compare both methods.

.. code-block:: python

   if rank == 0:
       plt.plot(w_saturation_standard, capillary_pressure, ".", label="Standard Method")
       plt.plot(w_saturation_contact_angle, capillary_pressure, ".", label="Contact Angle Method")
       plt.xlabel("Wetting Phase Saturation")
       plt.ylabel("Capillary Pressure")
       plt.legend()
       plt.savefig("examples/drainage_inkbottle/saturation_pressure_plot.png")
       plt.close()


.. image:: /_static/examples/drainage_inkbottle/saturation_pressure_plot.png
   :alt: Capillary pressure curves
   :class: only-light
   :align: center
   :width: 60%

Output
------

The expected output from a successful run is:

- :code:`image.pvti` and a folder :code:`image_proc` with two :code:`.vti` files which can be opened in **Paraview**
- :code:`saturation_pressure_plot.png`: Plot of capillary pressure vs. saturation.

The code used to generate the plots in this example is located at :code:`examples/drainage_inkbottle/plot_drainage_inkbottle.py` and must be run with :code:`pvpython`, ParaView's Python interpreter.


References
----------

.. bibliography::
   :style: unsrt
