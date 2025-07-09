# PMMoTo

![Tests](https://github.com/tmweigand/PMMoTo/actions/workflows/tests.yml/badge.svg)

![logo-1-text](https://github.com/tmweigand/PMMoTo/assets/68024672/5f667c8f-5498-4597-9af0-76fd6a9bc59a)

---

The Porous Media Morphology and Topology (PMMoTo) toolkit is an open-source Python library—backed by Cython and C++—for analyzing, modeling, and characterizing the structure of porous materials. The code is designed for performance on distributed memory systems using MPI. It includes fast connected components analysis, morphological operations (e.g., addition and subtraction), and Euclidean distance transforms. Output images are optimized for visualization in ParaView. PMMoTo also provides processing and analysis tools for reconstructing porous structures from molecular dynamics simulations.

---

## Website

https://tmweigand.github.io/PMMoTo/index.html

## Installation

**PMMoTo requires an MPI installation** (see https://tmweigand.github.io/PMMoTo/user_guide/installation.html)

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

### Development

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

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions, issues, or contributions, please open an issue or pull request on [GitHub](https://github.com/tmweigand/PMMoTo).
