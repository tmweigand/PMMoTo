==================
Tox-Based Testing
==================

This project uses ``tox`` to orchestrate testing, quality checks, and
coverage analysis across multiple Python versions and execution modes.
Using ``tox`` is the **recommended** way to run tests locally and is the
same mechanism used in continuous integration.

Overview
--------
``tox`` manages:
    - Multiple Python versions (3.10, 3.11, 3.12)
    - Serial and MPI-enabled test execution
    - Linting, formatting, static type checking, and documentation checks
    - Coverage aggregation across MPI ranks

All environments are defined in ``tox.ini``.

Test Matrix
-----------
The test suite is exercised across the following dimensions:

**Python versions**
    - Python 3.10
    - Python 3.11
    - Python 3.12

**Execution modes**
    - Serial unit tests
    - MPI-enabled tests using 1, 2, 4, and 8 processes

**Quality and analysis checks**
    - Ruff (linting)
    - Black (formatting)
    - mypy (static typing)
    - interrogate (docstring coverage)
    - pytest-cov (test coverage)

Running Tox
-----------
To run the full test and quality-check matrix:

.. code-block:: shell

   tox

This will:
    - Create isolated virtual environments for each Python version
    - Install all required dependencies
    - Run serial and MPI tests
    - Execute all configured quality checks

Running Individual Environments
--------------------------------

Specific environments can be run using ``-e``:

.. code-block:: shell

   tox -e py311      # tests with Python 3.11
   tox -e lint       # Ruff linting
   tox -e black      # formatting check
   tox -e mypy       # static type checking
   tox -e doccheck   # docstring coverage
   tox -e coverage   # coverage analysis

Unit and MPI Tests
------------------

The default Python test environments (``py310``, ``py311``, ``py312``)
run both serial unit tests and MPI-enabled tests using multiple process counts using the ``pytest-mpi`` plugin. Internally, ``tox`` invokes MPI tests with different numbers of ranks to validate correctness under parallel execution.

MPI Implementation Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different MPI implementations handle process oversubscription differently. To ensure consistent behavior across development machines, containers, and CI environments, the test runner:

    - Detects the MPI implementation at runtime
    - Automatically enables ``--oversubscribe`` when required (e.g., Open MPI)
    - Avoids oversubscription flags when not needed (e.g., MPICH)

This logic is handled internally and does not require user intervention.

Coverage Analysis
-----------------

Coverage is computed across both serial and MPI executions.

To run coverage analysis only:

.. code-block:: shell

   tox -e coverage

This environment will:
    - Run serial and MPI tests under coverage
    - Collect coverage data from all MPI ranks
    - Combine results into a single report
    - Enforce a minimum coverage threshold (80%)
    - Generate an HTML report in ``htmlcov/``

Linting and Formatting
----------------------

**Ruff (linting)**:

.. code-block:: shell

   tox -e lint

**Black (formatting)**:

.. code-block:: shell

   tox -e black

Static Type Checking
--------------------

Static type checking is performed using ``mypy`` with SciPy stubs:

.. code-block:: shell

   tox -e mypy

Documentation Coverage
----------------------

Docstring coverage is enforced using ``interrogate``:

.. code-block:: shell

   tox -e doccheck

A minimum docstring coverage threshold of 90% is required, excluding
selected auto-generated or external-interface modules.

Python Version Management
-------------------------

Running the full ``tox`` matrix requires Python 3.10–3.12 to be
available on the system. Installing and managing multiple Python
versions is documented separately.

See :doc:`python_versions` for instructions using ``pyenv``.

Continuous Integration
----------------------

The same ``tox`` environments are executed in GitHub Actions via the
``[gh-actions]`` mapping in ``tox.ini``. This ensures consistency
between local development and CI execution.
