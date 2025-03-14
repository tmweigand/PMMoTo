"""test_generate_rdf.py"""

from mpi4py import MPI
import numpy as np
import pytest
import pmmoto
import matplotlib.pyplot as plt

import gzip
from pmmoto.io import io_utils

dual_key_dict = {
    (1, 0.743425): 1,
    (1, 0.6797): 2,
    (3, 0.1716): 3,
    (3, -0.1565): 4,
    (3, -0.23): 5,
    (3, 0.014): 6,
    (3, -0.1956): 7,
    (4, -0.587509): 8,
    (5, 0.131): 9,
    (5, 0.10745): 10,
    (5, 0.1816): 11,
    (7, -0.398375): 12,
    (7, -0.4621): 13,
    (8, 0.23105): 14,
    (12, -0.5351): 15,
    (14, 0.4315): 16,
}


def read_lammps_atoms(input_file, label_map):
    """
    Read position of atoms from LAMMPS file
    atom_map must sync with LAMMPS ID
    """

    io_utils.check_file(input_file)

    if input_file.endswith(".gz"):
        domain_file = gzip.open(input_file, "rt")
    else:
        domain_file = open(input_file, "r", encoding="utf-8")

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
        elif 5 <= n_line <= 7:
            domain_data[n_line - 5, 0] = float(line.split(" ")[0])
            domain_data[n_line - 5, 1] = float(line.split(" ")[1])
        elif n_line >= 9:
            split = line.split(" ")

            a_type = int(split[2])
            charge = float(split[4])
            _type = label_map[(a_type, charge)]

            atom_type[count_atom] = _type
            for count, n in enumerate([5, 6, 7]):
                atom_position[count_atom, count] = float(split[n])  # x,y,z,atom_id

            count_atom += 1

    domain_file.close()

    return atom_position, atom_type, domain_data


def gen_radii(atom_ids, value):
    radii = {}
    for _id in atom_ids:
        radii[_id] = value
    return radii


@pytest.mark.skip
# @pytest.mark.mpi(min_size=8)
def generate_rdf():
    """
    Test for generating a radial distribution function from LAMMPS data
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    membrane_atom_label_file = "tests/test_data/LAMMPS/atom_map.txt"

    atom_labels_to_name = pmmoto.io.data_read.read_atom_map(membrane_atom_label_file)

    membrane_file = "tests/test_data/LAMMPS/membranedata.100005000.gz"
    water_file = "tests/test_data/LAMMPS/pressuredata.100005000.gz"

    membrane_positions, membrane_atom_type, domain_data = read_lammps_atoms(
        membrane_file, dual_key_dict
    )

    water_positions, water_atom_type, domain_data = (
        pmmoto.io.data_read.read_lammps_atoms(water_file)
    )

    # Ignore water "reservoirs"
    # domain_data[2] = [-150, 175]

    membrane_radii = gen_radii(membrane_atom_type, 2.75)
    water_radii = gen_radii(water_atom_type, 2.75)

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
        trim_within=True,
    )

    water = water.return_list(15)

    num_bins = 500

    bins = pmmoto.domain_generation.rdf.generate_bins(membrane_radii, num_bins)

    rdf_bins = pmmoto.domain_generation.rdf.generate_rdf(
        subdomain=sd, probe_atom_list=water, atoms=membrane, bins=bins
    )

    if rank == 0:
        # Set style for publication-quality figures
        # plt.style.use("seaborn")

        # Create figure with specific size
        plt.figure(figsize=(10, 6))

        # Plot RDFs for each atom type
        for label, rdf in rdf_bins.items():
            if sd.rank == 0:
                print(label)
            plt.plot(
                bins.bin_centers[label],
                rdf,
                label=f"Atom type {atom_labels_to_name[label]}",
                linewidth=2,
            )

        # Customize plot
        plt.xlabel("Distance (Å)", fontsize=12)
        plt.ylabel("Radial Distribution Function g(r)", fontsize=12)
        plt.title("Radial Distribution Functions by Atom Type", fontsize=14)
        plt.legend(frameon=True)
        plt.grid(True, alpha=0.3)

        # Set axis limits and ticks
        plt.xlim(left=0)
        plt.ylim(bottom=0)

        # Adjust layout to prevent label clipping
        plt.tight_layout()

        # Save figure with high DPI
        plt.savefig(f"data_out/rdf_all_types.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        # Also save individual plots
        for label, rdf in rdf_bins.items():
            plt.figure(figsize=(8, 5))
            plt.plot(bins.bin_centers[label], rdf, linewidth=2, color="navy")

            plt.xlabel("Distance (Å)", fontsize=12)
            plt.ylabel("g(r)", fontsize=12)
            plt.title(f"RDF for Atom Type {atom_labels_to_name[label]}", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xlim(left=0)
            plt.ylim(bottom=0)

            plt.tight_layout()
            plt.savefig(
                f"data_out/generate_rdf_{atom_labels_to_name[label]}.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    pmmoto.io.output.save_img_data_parallel(
        "data_out/test_lammps_subdomain", sd, np.zeros(sd.voxels)
    )
    pmmoto.io.output.save_particle_data(
        "data_out/test_lammps",
        sd,
        membrane.return_np_array(return_own=True, return_label=True),
    )

    pmmoto.io.output.save_particle_data(
        "data_out/test_water",
        sd,
        water.return_np_array(return_own=True, return_label=True),
    )


if __name__ == "__main__":
    generate_rdf()
