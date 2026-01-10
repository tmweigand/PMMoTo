Building the Documentation
==========================

PMMoTo documentation is built using `Sphinx <https://www.sphinx-doc.org/>`_.
API reference pages are generated automatically from the source code.

This page describes how to build the documentation locally for development
and preview purposes.

Prerequisites
-------------

Before building the documentation, ensure that:
    - You have a supported Python version installed
    - Project dependencies are installed
    - You have cloned the PMMoTo repository

All documentation dependencies are handled by the project and do not need
to be installed manually.

Local Documentation Build
-------------------------

PMMoTo provides a single script to generate API documentation and build
the HTML docs.

From the **root of the repository**, run:

.. code-block:: shell

   ./scripts/make_docs.sh

This script performs the following steps:

1. **Removes previously generated API files**
   in ``docs/api/`` to ensure a clean rebuild.
2. **Runs ``sphinx-apidoc``** on ``src/pmmoto`` to regenerate the API
   reference pages.
3. **Builds the HTML documentation** using Sphinx.

Output Location
---------------

After a successful build, the HTML documentation is located at:

.. code-block:: text

   docs/_build/html

Open ``docs/_build/html/index.html`` in a web browser to view the documentation
locally.

Updating Documentation
----------------------

If you modify:

- ``.rst`` documentation files
- Python docstrings
- Public APIs

re-run the build script:

.. code-block:: shell

   ./scripts/make_docs.sh

The API reference and rendered HTML will be regenerated automatically.

Notes for Contributors
----------------------

- The API documentation in ``docs/api/`` is **generated** and should not be
  edited manually.
- Documentation warnings are shown during the build. Contributors are
  encouraged to resolve warnings when modifying documentation.
- Documentation contributions follow the same workflow as code contributions
  (see :doc:`contributing`).
