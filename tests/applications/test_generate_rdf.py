"""test_generate_rdf.py"""

from mpi4py import MPI
import numpy as np
import pytest
import pmmoto


def gen_radii(atom_ids, value):
    radii = {}
    for _id in atom_ids:
        radii[_id] = value
    return radii


# @pytest.mark.mpi(min_size=8)
def generate_rdf():
    """
    Test for generating a radial distribution function from LAMMPS data
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    membrane_file = "tests/test_data/LAMMPS/membranedata.100005000.gz"
    water_file = "tests/test_data/LAMMPS/pressuredata.100005000.gz"

    membrane_positions, membrane_atom_type, domain_data = (
        pmmoto.io.data_read.read_lammps_atoms(membrane_file)
    )

    water_positions, water_atom_type, domain_data = (
        pmmoto.io.data_read.read_lammps_atoms(water_file)
    )

    # Ignore water "reservoirs"
    domain_data[2] = [-150, 175]

    membrane_radii = gen_radii(membrane_atom_type, 3.0)
    water_radii = gen_radii(water_atom_type, 3.0)

    sd = pmmoto.initialize(
        voxels=(100, 100, 100),
        box=domain_data,
        rank=rank,
        subdomains=(2, 2, 2),
        boundary_types=((2, 2), (2, 2), (2, 2)),
    )

    membrane = pmmoto.domain_generation.particles.initialize_atoms(
        sd,
        membrane_positions,
        membrane_radii,
        membrane_atom_type,
        by_type=True,
        add_periodic=True,
        set_own=True,
    )

    water = pmmoto.domain_generation.particles.initialize_atoms(
        sd,
        water_positions,
        water_radii,
        water_atom_type,
        by_type=True,
        set_own=True,
        trim=True,
    )

    water = water.return_list(16)

    num_bins = 10
    rdf_bins = pmmoto.domain_generation.rdf.generate_rdf(sd, water, membrane, num_bins)

    print(rdf_bins)

    # pmmoto.io.output.save_img_data_parallel(
    #     "data_out/test_lammps_subdomain", sd, np.zeros(sd.voxels)
    # )
    # pmmoto.io.output.save_particle_data(
    #     "data_out/test_lammps",
    #     sd,
    #     membrane.return_np_array(return_own=True, return_label=True),
    # )

    # pmmoto.io.output.save_particle_data(
    #     "data_out/test_water",
    #     sd,
    #     water.return_np_array(return_own=True, return_label=True),
    # )


if __name__ == "__main__":
    generate_rdf()
