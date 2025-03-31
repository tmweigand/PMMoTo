"""dataRead.py"""

import os
import gzip
import numpy as np

from . import io_utils

__all__ = [
    "read_sphere_pack_xyzr_domain",
    "read_r_lookup_file",
    "py_read_lammps_atoms",
    "read_lammps_atoms",
    "read_atom_map",
    "read_rdf",
]


def read_sphere_pack_xyzr_domain(input_file):
    """
    Read in sphere pack given in x,y,z,radius order including domain bounding box

    Input File Format:
        x_min x_max
        y_min y_max
        z_min z_max
        x1 y1 z1 r1
        x2 y2 z2 r2
        x3 y3 z3 r3
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

    domain_data = tuple(map(tuple, domain_data))

    return sphere_data, domain_data


def read_r_lookup_file(input_file, power=1):
    """
    Read in the radius lookup file for LAMMPS simulations

    Actually reading in sigma

    File is:
    Atom_ID epsilon sigma

    """
    io_utils.check_file(input_file)

    r_lookup_file = open(input_file, "r", encoding="utf-8")

    sigma = {}  # Lennard-Jones
    lookup_lines = r_lookup_file.readlines()

    for n_line, line in enumerate(lookup_lines):
        sigma_i = float(line.split(" ")[2])
        sigma[n_line + 1] = power * sigma_i

    r_lookup_file.close()

    return sigma


def py_read_lammps_atoms(input_file, include_mass=False):
    """
    Read position of atoms from LAMMPS file
    atom_map must sync with LAMMPS ID
    """

    io_utils.check_file(input_file)

    if input_file.endswith(".gz"):
        domain_file = gzip.open(input_file, "rt")
    else:
        domain_file = open(input_file, "r", encoding="utf-8")

    charges = {}

    lines = domain_file.readlines()
    domain_data = np.zeros([3, 2], dtype=np.double)
    count_atom = 0
    for n_line, line in enumerate(lines):
        if n_line == 1:
            time_step = float(line)
        elif n_line == 3:
            num_objects = int(line)
            atom_position = np.zeros([num_objects, 3], dtype=np.double)
            atom_type = np.zeros(num_objects, dtype=int)
            if include_mass:
                masses = np.zeros(num_objects, dtype=float)
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


def read_lammps_atoms(input_file, type_map=None):
    """
    Call to c++ read

    type_map (dict, optional): Mapping of (type, charge) pairs to new types
    Example: {(1, 0.4): 2, (1, -0.4): 3}
    """
    from . import _data_read

    positions, types, domain, timestep = _data_read.read_lammps_atoms(
        input_file, type_map
    )

    return positions, types, domain, timestep


def read_rdf(input_folder):
    """
    Read input folder containing radial distribution function data of the form

        radial distance, g(r), coordination number(r)

    Folder must contain file called `atom_map.txt`
    Files for all listed atoms of name 'atom_name'.rdf
    """

    # Check folder exists
    io_utils.check_folder(input_folder)

    # Check for atom_map.txt
    atom_map_file = input_folder + "atom_map.txt"
    io_utils.check_file(atom_map_file)

    atom_map = read_atom_map(atom_map_file)

    # Check rdf files found for all atoms
    atom_data = {}
    for label, atom_info in atom_map.items():
        atom_file = input_folder + atom_info["label"] + ".rdf"
        io_utils.check_file(atom_file)
        data = np.genfromtxt(atom_file)
        atom_data[label] = data

    return atom_map, atom_data


def read_atom_map(input_file):
    """
    Read in the atom mapping file which has the following format:
        atom_id element_name atom_name
    """
    # Check input file and proceed of exists
    io_utils.check_file(input_file)

    atom_file = open(input_file, "r", encoding="utf-8")
    atom_data = {}

    lines = atom_file.readlines()
    for line in lines:
        split = line.split(" ")
        element = split[1]
        label = split[2].split("\n")[0]
        atom_data[int(split[0])] = {"element": element, "label": label}

    return atom_data
