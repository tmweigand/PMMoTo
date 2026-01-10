Linux Testing with Docker
===========================

This project includes a Docker-based workflow to enable testing in a Linux
environment. This is primarily intended for development on macOS systems where
Linux-specific behavior (e.g., MPI, filesystem semantics, or build tooling)
cannot be tested natively.

Why Docker Is Used
--------------------

PMMoTo is developed primarily on macOS, but it targets Linux-based HPC and CI
environments. Docker provides a lightweight and reproducible way to:

- Test Linux-only dependencies and build behavior
- Validate MPI-related functionality in a Linux environment
- Reproduce CI failures locally
- Avoid maintaining a separate Linux machine or virtual machine

Using Docker ensures that tests closely match the environment used in GitHub
Actions and production deployments.

Building the Docker Image
-------------------------

From the project root, build the Docker image:

.. code-block:: shell

   docker build -t pmmoto-image .

This command:

- Uses the ``Dockerfile`` in the repository root
- Installs all required system and Python dependencies
- Creates a reusable Linux testing environment

Running the Docker Container
----------------------------

Start a container and mount the local project directory:

.. code-block:: shell

   docker run --name pmmoto-container -d \
     -v $(pwd):/home/mpitest/pmmoto \
     pmmoto-image

This will:

- Run the container in detached mode (``-d``)
- Mount the current working directory into the container
- Allow code changes on the host to be immediately visible inside the container

The container can be reused across multiple test runs.

Accessing the Container
-----------------------

To open an interactive shell inside the running container:

.. code-block:: shell

   docker exec -it pmmoto-container bash

Once inside, navigate to the mounted project directory:

.. code-block:: shell

   cd /home/mpitest/pmmoto

From here, you can run tests, build wheels, or execute MPI commands as you would
on a native Linux system.

Stopping and Cleaning Up
------------------------

Stop the container:

.. code-block:: shell

   docker stop pmmoto-container

Remove the container:

.. code-block:: shell

   docker rm pmmoto-container

If needed, remove the image:

.. code-block:: shell

   docker rmi pmmoto-image

Notes
-----

- Docker is intended for **local Linux testing only**, not production use.
- The mounted volume allows rapid iteration without rebuilding the image.
- If system dependencies change, rebuild the image to ensure consistency.
