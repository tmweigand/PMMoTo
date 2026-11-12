=======================================
Installation for Development
=======================================


There are two common ways to set up PMMoTo for development or testing:

1. **Editable install** (recommended for active development)

   .. code-block:: bash

      pip install -e .[dev]

   **Requirements and behavior:**

   - You must **first clone the PMMoTo repository** locally:

     .. code-block:: bash

        git clone https://github.com/tmweigand/PMMoTo.git
        cd pmmoto

   - Installs PMMoTo in **editable mode**, so changes to the source code
     (in ``src/pmmoto/``) are immediately available without reinstalling.
   - Installs all **development dependencies**, including tools for testing, linting,
     benchmarking, and documentation.

   Use this setup if you plan to **modify the code or contribute** to PMMoTo.

2. **Standard install with dev extras** (for testing or CI)

   .. code-block:: bash

      pip install pmmoto[dev]

   **Requirements and behavior:**

   - Installed directly from **PyPI**, no need to clone the repository.
   - Installs PMMoTo and all development dependencies, but **not in editable mode**.
   - Changes to the local source code will **not** take effect unless the package is reinstalled.

   Use this option if you only need to **run tests, coverage, or documentation builds**
   without modifying the source.
