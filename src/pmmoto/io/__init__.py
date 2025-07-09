"""Initialize the `io` subpackage for PMMoTo.

Provides data reading and output utilities accessible directly from `pmmoto.io`.
"""

from . import data_read, output

# Re-export selected functions from data_read
from .data_read import (
    read_sphere_pack_xyzr_domain,
    py_read_lammps_atoms,
    read_lammps_atoms,
    read_atom_map,
    read_rdf,
    read_binned_distances_rdf,
)

# Re-export selected functions from output
from .output import (
    save_particle_data,
    save_img,
    save_extended_img_data_parallel,
)

__all__ = [
    "data_read",
    "read_sphere_pack_xyzr_domain",
    "py_read_lammps_atoms",
    "read_lammps_atoms",
    "read_atom_map",
    "read_rdf",
    "read_binned_distances_rdf",
    "output",
    "save_particle_data",
    "save_img_data_serial",
    "save_img_data_parallel",
    "save_extended_img_data_parallel",
    "save_img",
]
