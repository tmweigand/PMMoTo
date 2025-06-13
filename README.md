# PMMoTo

![Tests](https://github.com/tmweigand/PMMoTo/actions/workflows/tests.yml/badge.svg)

![logo-1-text](https://github.com/tmweigand/PMMoTo/assets/68024672/5f667c8f-5498-4597-9af0-76fd6a9bc59a)

---

**PMMoTo** (Parallel Morphological Multiphase Toolkit) is an open-source, MPI-parallelized Python/C++ toolkit for simulating multiphase flow and morphological processes in complex porous media. PMMoTo provides efficient, scalable algorithms for generating, analyzing, and simulating pore-scale phenomena in 2D and 3D domains, with a focus on high-performance computing and extensibility.

---

## Features

- **Parallel domain decomposition** using MPI for distributed-memory simulations
- **Morphological drainage and imbibition** algorithms for capillary-dominated flow
- **Flexible porous media generation**: random fields, sphere packs, atomistic models, and more
- **Multiphase simulation**: support for multiple fluid phases and wettability
- **Distance transforms**: exact Euclidean distance transforms with periodic and distributed support
- **Capillary pressure–saturation curve computation** and other pore-scale analysis tools
- **Parallel VTK output** for visualization in ParaView and other tools
- **Modular, extensible design**: Python/C++ hybrid, easy to add new algorithms

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/tmweigand/PMMoTo.git
   cd PMMoTo
   ```

2. **Install dependencies:**

   - Python 3.8+
   - `numpy`
   - `mpi4py`
   - `matplotlib`
   - (Optional) `pytest` for testing

   Install with pip:

   ```bash
   pip install -r requirements.txt
   ```

3. **Build C/C++ extensions (required for performance):**
   ```bash
   python setup.py build_ext --inplace
   ```

---

## Usage

### Example: Morphological Drainage in an Inkbottle Geometry

```bash
mpirun -np 8 python examples/drainage_inkbottle.py
```

- Output images and data are saved in `examples/` or `data_out/`.
- Results can be visualized with ParaView or other VTK-compatible tools.

### Custom Porous Media

You can generate custom porous media using built-in generators (random, spheres, atoms) or your own. See `examples/drainage_inkbottle.py` and the `domain_generation` module for details.

---

## Directory Structure

```
src/pmmoto/
    core/               # Core parallel and utility routines (domain, subdomain, communication, utils)
    domain_generation/  # Porous media and multiphase domain generation
    filters/            # Morphological and distance transform algorithms
    io/                 # Input/output, VTK export, data readers
    analysis/           # Analysis tools (Minkowski functionals, etc.)
    particles/          # Particle and atom data structures
    examples/           # Example scripts and workflows
    tests/              # Unit and integration tests
```

---

## Testing

Run the test suite (requires `pytest` and `mpi4py`):

```bash
mpirun -np 8 python -m pytest tests/
```

---

## Documentation

- **API documentation:** See docstrings in source files.
- **Examples:** See the `examples/` directory for ready-to-run scripts.

---

## Citing

If you use PMMoTo in your research, please cite the relevant publication (add citation here if available).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions, issues, or contributions, please open an issue or pull request on [GitHub](https://github.com/tmweigand/PMMoTo).

---

**PMMoTo** — Parallel Morphological Multiphase Toolkit for Porous Media Simulation
