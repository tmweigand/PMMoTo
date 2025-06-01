![Tests](https://github.com/tmweigand/PMMoTo/actions/workflows/tests.yml/badge.svg)

![logo-1-text](https://github.com/tmweigand/PMMoTo/assets/68024672/5f667c8f-5498-4597-9af0-76fd6a9bc59a)

---

**PMMoto** is a parallel, morphological, multiphase pore-scale simulation toolkit for porous media research. It provides efficient, scalable algorithms for simulating XXX in complex geometries using MPI-based parallelization.

---

## Features

- **Parallel domain decomposition** using MPI for large-scale simulations
- **Morphological drainage and imbibition** algorithms
- Flexible support for custom porous media generation (e.g., sphere packs, molecular dynamics simulations)
- Capillary pressure–saturation curve computation
- Output in parallel VTK formats for visualization
- Modular design for extensibility

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

3. **Build any required C/C++ extensions** (if needed):
   ```bash
   python setup.py build_ext --inplace
   ```

---

## Usage

### Running a Drainage Simulation

Example: Morphological drainage in an inkbottle geometry

```bash
mpirun -np 8 python examples/drainage_inkbottle.py
```

- Output images and data will be saved in the `examples/` or `data_out/` directories.
- Results can be visualized with ParaView or other VTK-compatible tools.

### Custom Porous Media

You can use or implement your own porous media generator and pass it to the simulation driver. See `examples/drainage_inkbottle.py` for details.

---

## Directory Structure

```
src/pmmoto/
    core/               # Core parallel and utility routines
    domain_generation/  # Porous media generation functions
    filters/            # Morphological and equilibrium algorithms
    io/                 # Input/output and VTK export
    analysis/           # Analysis tools
    examples/           # Example scripts
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

If you use PMMoto in your research, please cite the relevant publication (add citation here if available).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions, issues, or contributions, please open an issue or pull request on [GitHub](https://github.com/tmweigand/PMMoTo).

---

**PMMoto** — Parallel Morphological Multiphase Toolkit for Porous Media Simulation
