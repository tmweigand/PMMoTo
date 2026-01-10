Testing
=======

This project supports two testing workflows:
    - Running tests directly with a single Python interpreter
    - Running the full multi-version and quality-check matrix using ``tox``

Single-Python Workflow
----------------------

Run serial unit tests:

.. code-block:: shell

   pytest tests/

MPI tests are marked with the pytest marker ``@pytest.mark.mpi()`` and require the
``pytest-mpi`` plugin. Only tests decorated with this marker are executed when running MPI tests.
For example, an MPI test function can be defined like this:

.. code-block:: python

   import pytest

   @pytest.mark.mpi()
   def test_parallel_function():
       ...

To run MPI tests with 8 processes:

.. code-block:: shell

   mpiexec -n 8 pytest tests/ --only-mpi

Notes:

- The ``--only-mpi`` flag is implemented via ``pytest-mpi``.
- Serial tests (without the marker) are skipped when using ``--only-mpi``.

.. note::
  
  On some shared systems, Open MPI may refuse to launch MPI jobs if the requested number of
  processes exceeds available CPU cores. In these cases, the ``--oversubscribe`` flag is required.
  The ``tox`` workflow handles this automatically, so you usually only need to worry about it
  when running ``mpiexec`` manually.


Tox Workflow (Recommended)
--------------------------

To run the full test matrix:

.. code-block:: shell

   tox

This runs:
    - Unit and MPI tests across all supported Python versions
    - Linting, formatting, type checking, and coverage

See :doc:`tox` for details on the test matrix and available environments.

.. note::

   Running ``tox`` requires Python 3.10-3.12 for complete testing. See
   :doc:`python_versions` for installation instructions.