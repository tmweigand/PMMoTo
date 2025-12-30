=============================
Managing Python Versions
=============================

Running the full ``tox`` test matrix requires multiple Python versions
to be available on the local system. This project currently tests
against:

    - Python 3.10
    - Python 3.11
    - Python 3.12

The recommended and supported way to manage multiple Python versions is
to use ``pyenv``.

Why pyenv?
----------

The built-in ``venv`` module can only create virtual environments from
an already-installed Python interpreter. It cannot install or manage
multiple Python versions.

``pyenv`` allows multiple Python interpreters to be installed side by
side and selected on a per-project basis. ``tox`` then uses these
interpreters to automatically create isolated virtual environments.

macOS (OS X)
------------

Installing pyenv
^^^^^^^^^^^^^^^^

Install ``pyenv`` using Homebrew:

.. code-block:: shell

   brew install pyenv

Add the following to your shell configuration file
(e.g. ``~/.zshrc`` or ``~/.bashrc``):

.. code-block:: shell

   export PYENV_ROOT="$HOME/.pyenv"
   export PATH="$PYENV_ROOT/bin:$PATH"
   eval "$(pyenv init -)"

Restart your shell and verify the installation:

.. code-block:: shell

   pyenv --version

Installing Python Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^

Install the required Python versions:

.. code-block:: shell

   pyenv install 3.10.14
   pyenv install 3.11.9
   pyenv install 3.12.3

Exact patch versions are not critical, but should correspond to the
``py310``, ``py311``, and ``py312`` environments defined in ``tox.ini``.

Linux
-----

Installing pyenv
^^^^^^^^^^^^^^^^

On Linux, ``pyenv`` builds Python from source and requires several
system dependencies.

Install dependencies (example for Debian/Ubuntu):

.. code-block:: shell

   sudo apt update
   sudo apt install -y \
       build-essential \
       curl \
       git \
       libssl-dev \
       zlib1g-dev \
       libbz2-dev \
       libreadline-dev \
       libsqlite3-dev \
       libncursesw5-dev \
       xz-utils \
       tk-dev \
       libxml2-dev \
       libxmlsec1-dev \
       libffi-dev \
       liblzma-dev

Install ``pyenv`` using the official installer:

.. code-block:: shell

   curl https://pyenv.run | bash

Add the following to your shell configuration file
(e.g. ``~/.bashrc`` or ``~/.zshrc``):

.. code-block:: shell

   export PYENV_ROOT="$HOME/.pyenv"
   export PATH="$PYENV_ROOT/bin:$PATH"
   eval "$(pyenv init -)"

Restart your shell and verify the installation:

.. code-block:: shell

   pyenv --version

Installing Python Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^

Install the required Python versions:

.. code-block:: shell

   pyenv install 3.10.14
   pyenv install 3.11.9
   pyenv install 3.12.3

Project Configuration
---------------------

From the root of the repository, configure the local Python versions:

.. code-block:: shell

   pyenv local 3.10.14 3.11.9 3.12.3

This creates a ``.python-version`` file that ensures ``tox`` can locate
all required interpreters.

Verify that the interpreters are available:

.. code-block:: shell

   which python3.10
   which python3.11
   which python3.12

All paths should resolve to ``~/.pyenv/versions/...``.

Next Steps
----------

Once the required Python versions are installed, ``tox`` can be used to
run the full test and quality-check matrix.

See :doc:`tox` for details.
