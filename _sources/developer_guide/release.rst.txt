Creating and Uploading Releases to PyPI
========================================

This guide describes how PMMoTo releases are built and published to PyPI using
the GitHub Actions workflow **Build and Release**. Releases are fully automated
and triggered by pushing a version tag to the repository.

Prerequisites
-------------

Before creating a release, ensure:

- You have push access to the PMMoTo GitHub repository.
- The ``PYPI_API_TOKEN`` secret is configured in the GitHub repository settings.
- The package version has been updated according to semantic versioning.

Versioning
----------

PMMoTo follows `Semantic Versioning <https://semver.org/>`__:

- **Major version**: incompatible API changes
- **Minor version**: backward-compatible feature additions
- **Patch version**: backward-compatible bug fixes

Example version:

``1.2.3``

- ``1`` → major version
- ``2`` → minor version
- ``3`` → patch version

Git tags must be prefixed with ``v``:

``v1.2.3``

Updating the Version
--------------------

Before tagging a release, update the version number in ``pyproject.toml``

Commit the version change to the main branch:

.. code-block:: shell

   git add pyproject.toml
   git commit -m "Bump version to 1.2.3"
   git push

Creating a Release
------------------

Releases are triggered by pushing a version tag:

.. code-block:: shell

   git tag v1.2.3
   git push origin v1.2.3

This will automatically start the **Build and Release** GitHub Actions workflow.

What the Workflow Does
----------------------

The GitHub Actions workflow performs the following steps:

1. **Build wheels** using ``cibuildwheel`` on:
   - Ubuntu
   - macOS (Intel)
   - macOS (Apple Silicon)

2. **Build a source distribution (sdist)** on Linux.

3. **Upload all artifacts** to PyPI using the official PyPA publish action.

No manual steps are required after pushing the tag.

Workflow Trigger
----------------

The workflow is triggered by version tags matching:

.. code-block:: yaml

   on:
     push:
       tags:
         - "v*.*.*"

Publishing to PyPI
------------------

Publishing is handled automatically using:

- ``pypa/cibuildwheel`` for wheels
- ``python -m build`` for the sdist
- ``pypa/gh-action-pypi-publish`` for uploading

Authentication is done using the ``PYPI_API_TOKEN`` GitHub secret.

Verifying the Release
-----------------------

After the workflow completes successfully, verify the release at:

`https://pypi.org/project/pmmoto/ <https://pypi.org/project/pmmoto/>`__

You can also inspect the workflow logs in GitHub Actions to confirm that wheels
and the source distribution were uploaded correctly.

Notes
-----

- Do **not** upload releases manually using ``twine``.
- All official releases should be created via Git tags.
- If a release fails, fix the issue, bump the version, and create a new tag.
