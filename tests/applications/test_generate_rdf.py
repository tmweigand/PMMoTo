"""test_generate_rdf.py"""

import numpy as np
import pytest
import pmmoto


def test_generate_rdf():
    """
    Test for generating a radial distribution function from LAMMPS data
    """

    membrane_file = "tests/test_data/LAMMPS/membranedata.100005000.gz"
    water_file = "tests/test_data/LAMMPS/pressuredata.100005000.gz"

    membrane_positions, membrane_atom_type, domain_data = (
        pmmoto.io.data_read.read_lammps_atoms(membrane_file)
    )

    # Want to examine by atom type. Need to implement better approach
    atom_types = np.unique(membrane_atom_type)

    ind = np.where(membrane_atom_type == atom_types[0])
    membrane_positions = membrane_positions[ind]

    num_membrane_atoms = membrane_positions.shape[0]
    atom_radii = np.ones([num_membrane_atoms, 1]) * 10
    membrane = np.concatenate((membrane_positions, atom_radii), axis=1)

    water_positions, water_atom_type, domain_data = (
        pmmoto.io.data_read.read_lammps_atoms(water_file)
    )

    num_water_atoms = water_positions.shape[0]
    atom_radii = np.ones([num_water_atoms, 1]) * 10
    water = np.concatenate((water_positions, atom_radii), axis=1)

    # Ignore water "reservoirs"
    domain_data[2] = [-150, 175]

    sd = pmmoto.initialize(
        voxels=(100, 100, 100),
        box=domain_data,
        subdomains=(2, 2, 2),
        boundary_types=((2, 2), (2, 2), (2, 2)),
    )

    membrane = pmmoto.domain_generation.particles.initialize(
        sd, membrane, add_periodic=True
    )

    water = pmmoto.domain_generation.particles.initialize(sd, water)

    pmmoto.io.output.save_img_data_parallel(
        "data_out/test_lammps_subdomain", sd, np.zeros(sd.voxels)
    )
    pmmoto.io.output.save_particle_data(
        "data_out/test_lammps", sd, membrane.return_np_array(return_own=True)
    )

    pmmoto.io.output.save_particle_data(
        "data_out/test_water", sd, water.return_np_array(return_own=True)
    )
