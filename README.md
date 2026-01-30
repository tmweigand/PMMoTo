# PMMoTo

[![Documentation](https://img.shields.io/badge/Documentation-PMMoTo-blue?style=for-the-badge)](https://tmweigand.github.io/PMMoTo/)

![Tests](https://github.com/tmweigand/PMMoTo/actions/workflows/tests.yml/badge.svg) [![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?style=flat-square)](https://www.python.org/) [![PyPI](https://img.shields.io/pypi/v/pmmoto?style=flat-square)](https://pypi.org/project/pmmoto/) [![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

![logo-1-text](https://github.com/tmweigand/PMMoTo/assets/68024672/5f667c8f-5498-4597-9af0-76fd6a9bc59a)

---

The Porous Media Morphology and Topology (PMMoTo) toolkit is an open-source Python library for analyzing, modeling, and characterizing the structure of porous materials. Built with Cython and C++ and designed for distributed memory systems using MPI, PMMoTo includes parallelized implementations of connected components analysis, morphological operations (e.g., dilation and erosion), and Euclidean distance transforms. Unlike many tools that focus on image-based analysis, PMMoTo is specifically designed for synthetically generated porous media, such as sphere packings and molecular dynamics simulations.

---

## Website

PMMoTo Website: [https://tmweigand.github.io/PMMoTo/](https://tmweigand.github.io/PMMoTo/)

## Installation

**PMMoTo requires MPI** (see https://tmweigand.github.io/PMMoTo/user_guide/installation.html)

### Standard

1. **Clone the repository:**

   ```bash
   git clone https://github.com/tmweigand/PMMoTo.git
   cd PMMoTo
   ```

2. **Install the package:**

   ```
   pip install .
   ```

### Development - including testing

For detailed instructions, see https://tmweigand.github.io/PMMoTo/developer_guide/installation.html

1. **Clone the repository:**

   ```bash
   git clone https://github.com/tmweigand/PMMoTo.git
   cd PMMoTo
   ```

2. Install in editable mode with dev dependencies:
   ```bash
   pip install -e .[dev]
   ```

---

## Citing

If you use PMMoTo in your research, please cite the relevant publication (too come).

---

## Community Guidelines

Contributions and community feedback is welcomed!

### Contributing

- Fork the repo and create a branch.
- Make your changes and submit a pull request.
- Ensure tests pass.

### Reporting Issues

- Use [GitHub Issues](https://github.com/tmweigand/PMMoTo/issues).
- Include details: steps to reproduce, expected vs. actual behavior, and system info.

### Support

- See the [docs](https://tmweigand.github.io/PMMoTo) and [examples](https://tmweigand.github.io/PMMoTo/examples).
- For questions, open a [discussion](https://github.com/tmweigand/PMMoTo/discussions) or file an [issue](https://github.com/tmweigand/PMMoTo/issues).

## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/tmweigand/PMMoTo/blob/main/LICENSE) for details.

---

## Contact

For questions, issues, or contributions, please open an issue or pull request on [GitHub](https://github.com/tmweigand/PMMoTo).
