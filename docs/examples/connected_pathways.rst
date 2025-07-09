Connected Pathways in Random Porous Media
=========================================

This example demonstrates how to identify connected pathways and isolated pores through a **connected components** analysis.

To run this example:

.. code-block:: bash

   mpirun -np 4 python examples/connected_pathways/connected_pathways.py

Step 1: Import Modules
----------------------

.. code-block:: python

   from mpi4py import MPI
   import numpy as np
   import pmmoto


Step 2: Domain Setup
--------------------

Initialize the domain and specify the inlet and outlet faces. For this example, we will analyze a porous structure that is thin in the z-dimension to allow for easier visualization. 

.. code-block:: python

   import pmmoto
   voxels = (2000, 2000, 10)
   box = ((0, 2000), (0, 2000), (0, 30))
   inlet = ((True, False), (False, False), (False, False))
   outlet = ((False, True), (False, False), (False, False))

   subdomains = (2, 2, 1)

   sd = pmmoto.initialize(
        voxels, rank=rank, subdomains=subdomains, box=box, inlet=inlet, outlet=outlet
   )

The domain decomposed into four subdomains is shown below. 

.. image:: /_static/examples/connected_pathways/subdomains.png
   :alt: Domain
   :class: only-light
   :align: center
   :width: 60%

Step 3: Generate Random Porous Media
------------------------------------

For the porous media, a smoothed random binary image is used.

.. code-block:: python

   img = pmmoto.domain_generation.gen_img_smoothed_random_binary(
       sd.domain.voxels, p_zero=0.5, smoothness=10, seed=8
   )

.. image:: /_static/examples/connected_pathways/pore_space.png
   :alt: Domain
   :class: only-light
   :align: center
   :width: 60%

Step 4: Domain Decomposition
----------------------------

While memory inefficient and should be avoided, PMMoTo is able to read an entire image and transfer ownership to the subdomain (i.e., decomposition).

.. code-block:: python

   sd, img_sd = pmmoto.domain_generation.deconstruct_img(sd, img, subdomains, rank)


Step 5: Label Connected Components
----------------------------------

A connected components analysis yields a labeled image where voxels of the same label are connected. The default (and currently only) option in PMMoTo is 26-connected (i.e., all faces, edges, and corners). Others include 18-connected (i.e., faces and edges) and 8-connected (i.e., faces). 

.. code-block:: python

   cc, label_count = pmmoto.filters.connected_components.connect_components(img_sd, sd)

.. image:: /_static/examples/connected_pathways/pore_space_labels.png
   :alt: Domain
   :class: only-light
   :align: center
   :width: 60%

Step 6: Inlet/Outlet Connectivity
---------------------------------

The labeled voxels can be assess to determine if the are connected to the inlet and/or outlet of the porous media. A voxel is connected to the inlet/outlet if the voxel lies on the inlet/outlet face. If a labeled set of voxels is location on both the inlet and the outlet, the labeled set is a connected pathway. If a labeled set of voxels is not connected to an inlet or outlet, the labeled set is isolated. 

.. code-block:: python

   inlet_img = pmmoto.filters.connected_components.inlet_connected_img(sd, img_sd)
   outlet_img = pmmoto.filters.connected_components.outlet_connected_img(sd, img_sd)
   inlet_outlet_img = pmmoto.filters.connected_components.inlet_outlet_connected_img(sd, img_sd)
   isolated_img = 

Inlet-Connected Voxels
-----------------------

These voxels are connected to the **inlet**:

.. image:: /_static/examples/connected_pathways/inlet_pore_space.png
   :alt: Inlet-connected domain
   :class: only-light
   :align: center
   :width: 60%


Outlet-Connected Voxels
------------------------

These voxels are connected to the **outlet**:

.. image:: /_static/examples/connected_pathways/outlet_pore_space.png
   :alt: Outlet-connected domain
   :class: only-light
   :align: center
   :width: 60%


Connected Pathway Voxels (Inlet and Outlet)
---------------------------------

These voxels are connected to **both** the inlet and outlet (i.e., a connected path):

.. image:: /_static/examples/connected_pathways/inlet_outlet_pore_space.png
   :alt: Fully connected domain
   :class: only-light
   :align: center
   :width: 60%

Isolated Voxels 
---------------------------------

These voxels are connected to **neither** the inlet and outlet:

.. image:: /_static/examples/connected_pathways/isolated_pore_space.png
   :alt: Domain
   :class: only-light
   :align: center
   :width: 60%


Step 7: Save Outputs
--------------------

.. code-block:: python

    pmmoto.io.output.save_img(
        "examples/connected_pathways/image",
        sd,
        img_sd,
        additional_img={
            "cc": cc,
            "inlet_img": inlet_img,
            "outlet_img": outlet_img,
            "inlet_outlet_img": inlet_outlet_img,
            "isolated_img": isolated_img,
        },
    )


Output
------

The expected output from a successful run is:

- :code:`image.pvti` and a folder :code:`image_proc` with eight :code:`.vti` files which can be opened in **Paraview**

The code used to generate the plots in this example is located at :code:`examples/connected_pathways/plot_connected_pathways.py` and must be run with :code:`pvpython`, ParaView's Python interpreter.




---