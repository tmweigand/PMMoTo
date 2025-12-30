======================
Contributing to PMMoTo
======================

Thank you for your interest in contributing to PMMoTo! Community contributions are welcome and encouraged. This guide describes the recommended workflow for contributing code, documentation, or tests.

Getting Started
---------------

1. **Fork the repository**

   Fork the PMMoTo repository on GitHub to your own account.

2. **Clone your fork**

   .. code-block:: shell

      git clone https://github.com/your-username/pmmoto.git
      cd pmmoto

3. **Set up a development environment**

   Follow the instructions in :doc:`installation`.
   Details on supported Python versions and platform-specific setup are documented in :doc:`python_versions`.

4. **Create a feature branch**

   Always create a new branch for your work:

   .. code-block:: shell

      git checkout -b feature/your-feature-name

Making Changes
--------------

- Keep changes focused and logically scoped.
- Follow existing coding style and project conventions.
- Add or update tests when modifying functionality.
- Update documentation where relevant.

If you are unsure about design choices or scope, opening an issue for discussion before implementing large changes is encouraged.

Testing and Quality Checks
--------------------------

Before submitting a contribution, run the full test and quality-check suite:

.. code-block:: shell

   tox

This runs the same checks enforced in continuous integration, including tests, linting, and coverage.

Submitting Your Contribution
-----------------------------

1. **Commit your changes**

   Write clear, descriptive commit messages:

   .. code-block:: shell

      git commit -m "Add feature: concise description"

2. **Push to your fork**

   .. code-block:: shell

      git push origin feature/your-feature-name

3. **Open a pull request**

   Create a pull request against the main PMMoTo repository and describe:
    - what the change does
    - why it is needed
    - any relevant issues it addresses

Maintainers may request revisions before merging.

Code of Conduct
---------------

All contributors are expected to follow the project’s :doc:`code_of_conduct` in all interactions.

Thank you for helping improve PMMoTo!
