"""dataRead.py

Provides functions for reading various simulation and atomistic data formats
used in PMMoTo, including sphere packs, LAMMPS files, atom maps, and RDF data.
"""

import gzip
import numpy as np
from numpy.typing import NDArray
from . import _data_read
from . import io_utils
from ..domain_generation import rdf
from ..analysis import bins

__all__ = [
    "read_sphere_pack_xyzr_domain",
    "py_read_lammps_atoms",
    "read_lammps_atoms",
    "read_atom_map",
    "read_rdf",
    "read_binned_distances_rdf",
]


def read_sphere_pack_xyzr_domain(
    input_file: str,
) -> tuple[NDArray[np.double], tuple[tuple[float, float], ...]]:
    """Read a sphere pack file with x, y, z, radius and domain bounding box.

    Input File Format:
        x_min x_max
        y_min y_max
        z_min z_max
        x1 y1 z1 r1
        x2 y2 z2 r2
        ...

    Args:
        input_file (str): Path to the input file.

    Returns:
        tuple: (sphere_data, domain_data)
            - sphere_data (np.ndarray): Array of shape (N, 4) for sphere positions
                                        and radii.
            - domain_data (tuple): Domain bounding box as ((x_min, x_max), ...).

    """
    # Check input file and proceed of exists
    io_utils.check_file(input_file)

    domain_file = open(input_file, "r", encoding="utf-8")
    lines = domain_file.readlines()
    num_spheres = len(lines) - 3

    sphere_data = np.zeros([num_spheres, 4], dtype=np.double)
    domain_data = np.zeros([3, 2], dtype=np.double)

    count_sphere = 0
    for n_line, line in enumerate(lines):
        if n_line < 3:  # Grab domain size
            domain_data[n_line, 0] = float(line.split(" ")[0])
            domain_data[n_line, 1] = float(line.split(" ")[1])
        else:  # Grab sphere
            try:
                for n in range(0, 4):
                    sphere_data[count_sphere, n] = float(line.split(" ")[n])
            except ValueError:
                for n in range(0, 4):
                    sphere_data[count_sphere, n] = float(line.split("\t")[n])
            count_sphere += 1

    domain_file.close()

    domain_box: tuple[tuple[float, float], ...] = tuple(map(tuple, domain_data))

    return sphere_data, domain_box


def py_read_lammps_atoms(
    input_file: str, include_mass: bool = False
) -> (
    tuple[NDArray[np.double], NDArray[np.uint8], NDArray[np.double], NDArray[np.double]]
    | tuple[NDArray[np.double], NDArray[np.uint8], NDArray[np.double]]
):
    """Read atom positions from a LAMMPS file.

    Args:
        input_file (str): Path to the LAMMPS file.
        include_mass (bool, optional): Whether to include mass data.

    Returns:
        tuple: (atom_position, atom_type, [masses,] domain_data)
            - atom_position (np.ndarray): Atom positions.
            - atom_type (np.ndarray): Atom types.
            - masses (np.ndarray, optional): Atom masses if include_mass is True.
            - domain_data (np.ndarray): Domain bounding box.

    """
    io_utils.check_file(input_file)

    if input_file.endswith(".gz"):
        domain_file = gzip.open(input_file, "rt")
    else:
        domain_file = open(input_file, "r", encoding="utf-8")

    charges: dict[int, list[float]] = {}

    lines = domain_file.readlines()
    domain_data = np.zeros([3, 2], dtype=np.double)
    count_atom = 0
    for n_line, line in enumerate(lines):
        if n_line == 1:
            _ = float(line)  # Time
        elif n_line == 3:
            num_objects = int(line)
            atom_position = np.zeros([num_objects, 3], dtype=np.double)
            atom_type = np.zeros(num_objects, dtype=np.uint8)
            if include_mass:
                masses = np.zeros(num_objects, dtype=np.double)
        elif 5 <= n_line <= 7:
            domain_data[n_line - 5, 0] = float(line.split(" ")[0])
            domain_data[n_line - 5, 1] = float(line.split(" ")[1])
        elif n_line >= 9:
            split = line.split(" ")

            type = int(split[2])
            atom_type[count_atom] = type
            charge = float(split[4])
            if type in charges:
                if charge not in charges[type]:
                    charges[type].append(charge)
            else:
                charges[type] = [charge]

            if include_mass:
                masses[count_atom] = float(split[3])

            for count, n in enumerate([5, 6, 7]):
                atom_position[count_atom, count] = float(split[n])  # x,y,z,atom_id

            count_atom += 1

    domain_file.close()

    if include_mass:
        return atom_position, atom_type, masses, domain_data
    else:
        return atom_position, atom_type, domain_data


def read_lammps_atoms(
    input_file: str, type_map: None | dict[tuple[int, float], int] = None
) -> tuple[NDArray[np.double], NDArray[np.uint8], NDArray[np.double], float]:
    """Read atom positions and types from a LAMMPS file using C++ backend.

    Args:
        input_file (str): Path to the LAMMPS file.
        type_map (dict, optional): Mapping of (type, charge) pairs to new types.
            Example: {(1, 0.4): 2, (1, -0.4): 3}

    Returns:
        tuple: (positions, types, domain, timestep)

    """
    positions, types, domain, timestep = _data_read.read_lammps_atoms(
        input_file, type_map
    )

    return positions, types, domain, timestep


def read_atom_map(input_file: str) -> dict[int, dict[str, str]]:
    """Read the atom mapping file.

    File Format:
        atom_id, element_name, atom_name

    Args:
        input_file (str): Path to the atom map file.

    Returns:
        dict: Mapping from atom ID to element and atom name.

    """
    # Check input file and proceed of exists
    io_utils.check_file(input_file)

    atom_file = open(input_file, "r", encoding="utf-8")
    atom_data = {}

    lines = atom_file.readlines()
    for line in lines:
        split = line.split(" ")
        element = split[1]
        atom_name = split[2].split("\n")[0]
        atom_data[int(split[0])] = {"element": element, "name": atom_name}

    return atom_data


def read_rdf(input_folder: str) -> tuple[dict[int, dict[str, str]], dict[int, rdf.RDF]]:
    """Read a folder containing radial distribution function (RDF) data.

    Folder must contain:
        - atom_map.txt
        - Files for all listed atoms named 'atom_name'.rdf

    Each .rdf file format:
        radial distance, rdf(r)

    Args:
        input_folder (str): Path to the folder.

    Returns:
        tuple: (atom_map, rdf_out)
            - atom_map (dict): Atom mapping.
            - rdf_out (dict): Mapping from atom ID to RDF object.

    """
    # Check folder exists
    io_utils.check_folder(input_folder)

    # Check for atom_map.txt
    atom_map_file = input_folder + "atom_map.txt"
    io_utils.check_file(atom_map_file)

    atom_map = read_atom_map(atom_map_file)

    # Check rdf files found for all atoms
    rdf_out = {}
    for _id, atom_info in atom_map.items():
        atom_file = input_folder + atom_info["name"] + ".rdf"
        io_utils.check_file(atom_file)
        data = np.genfromtxt(atom_file)
        rdf_out[_id] = rdf.RDF(
            name=atom_info["name"],
            atom_id=_id,
            radii=data[:, 0],
            rdf=data[:, 1],
        )

    return atom_map, rdf_out


def read_binned_distances_rdf(
    input_folder: str,
) -> tuple[dict[int, dict[str, str]], dict[int, rdf.RDF]]:
    """Read a folder containing binned RDF data.

    Folder must contain:
        - atom_map.txt
        - Files for all listed atoms named 'atom_name'.rdf

    Each .rdf file format:
        radial distance, rdf(r)

    Args:
        input_folder (str): Path to the folder.

    Returns:
        tuple: (atom_map, rdf_out)
            - atom_map (dict): Atom mapping.
            - rdf_out (dict): Mapping from atom ID to RDF object.

    """
    # Check folder exists
    io_utils.check_folder(input_folder)

    # Check for atom_map.txt
    atom_map_file = input_folder + "atom_map.txt"
    io_utils.check_file(atom_map_file)

    atom_map = read_atom_map(atom_map_file)

    # Check rdf files found for all atoms
    rdf_out = {}
    for _id, atom_info in atom_map.items():
        atom_file = input_folder + atom_info["name"] + ".rdf"
        io_utils.check_file(atom_file)
        _bins, binned_distances = np.genfromtxt(atom_file, skip_header=0, unpack=True)

        bin = bins.Bin(
            start=_bins[0],
            end=_bins[-1],
            num_bins=len(_bins),
            name=atom_info["name"],
            values=binned_distances,
        )
        rdf_out[_id] = rdf.RDF(
            name=atom_info["name"],
            atom_id=_id,
            radii=_bins,
            rdf=bin.generate_rdf(),
        )

    return atom_map, rdf_out
