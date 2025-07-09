"""Example: Generate the pore size distribution of a sphere packing."""

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import pmmoto

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def initialize_domain(voxels: tuple[int, ...]):
    """Initialize the membrane domain"""
    subdomains = (2, 2, 2)

    # Full domain with reservoirs
    box = [
        [0.0, 176],
        [0.0, 176],
        [-100, 100],
    ]

    sd = pmmoto.initialize(
        voxels=voxels,
        box=box,
        rank=rank,
        subdomains=subdomains,
        boundary_types=(
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        ),
        verlet_domains=(20, 20, 20),
        inlet=((False, False), (False, False), (True, False)),
        outlet=((False, False), (False, False), (False, True)),
    )

    return sd


def determine_uff_radii(atom_folder: str, radius: float):
    """Collect the radii given a pmf cutoff"""
    atom_map = pmmoto.io.data_read.read_atom_map(atom_folder + "atom_map.txt")
    radii = {}
    for atom_id, atom_data in atom_map.items():
        radii[atom_id] = (
            list(pmmoto.particles.uff_radius(atom_names=atom_data["element"]).values())[
                0
            ]
            + radius
        )

    return radii


def determine_pmf_radii(atom_folder, pmf_value):
    """Collect the radii given a pmf cutoff"""
    _, rdf = pmmoto.io.data_read.read_binned_distances_rdf(atom_folder)

    bounded_rdf = {}
    for _id, _rdf in rdf.items():
        bounded_rdf[_id] = pmmoto.domain_generation.rdf.BoundedRDF.from_rdf(
            _rdf, eps=1.0e-3
        )

    pmf_radii = {}
    for atom_id, _rdf in bounded_rdf.items():
        pmf_radii[atom_id] = _rdf.interpolate_radius_from_pmf(pmf_value)

    return pmf_radii


def compare_radii(pmf_value, subdomain, membrane_file):
    """Generate plots for comparing approaches"""
    # This maps from from the lammps input file id and charge
    # to a unique id. Atom_map.txt relates ids to atom types
    atom_id_charge_map = {
        (1, 0.6797): 1,
        (1, 0.743425): 2,
        (3, -0.23): 3,
        (3, -0.1956): 4,
        (3, -0.1565): 5,
        (3, 0.014): 6,
        (3, 0.1716): 7,
        (4, -0.587509): 8,
        (5, 0.10745): 9,
        (5, 0.131): 10,
        (5, 0.1816): 11,
        (7, -0.4621): 12,
        (7, -0.398375): 13,
        (8, 0.23105): 14,
        (12, -0.5351): 15,
        (14, 0.4315): 16,
    }

    columbic_water = 1.4

    # bounded_rdf = generate_bounded_rdf()
    atom_folder = "examples/md_porous_media/rdf_bins/"
    rdf_radii = determine_pmf_radii(atom_folder, pmf_value)
    uff_radii = determine_uff_radii(atom_folder, columbic_water)

    rdf_pm = pmmoto.domain_generation.gen_pm_atom_file(
        subdomain=subdomain,
        lammps_file=membrane_file,
        atom_radii=rdf_radii,
        type_map=atom_id_charge_map,
        kd=False,
        add_periodic=True,
    )

    rdf_pm.img = pmmoto.filters.morphological_operators.dilate(
        subdomain, rdf_pm.img, columbic_water
    )

    uff_pm = pmmoto.domain_generation.gen_pm_atom_file(
        subdomain=subdomain,
        lammps_file=membrane_file,
        atom_radii=uff_radii,
        type_map=atom_id_charge_map,
        kd=False,
        add_periodic=True,
    )

    uff_pm.img = pmmoto.filters.morphological_operators.dilate(
        subdomain, uff_pm.img, columbic_water
    )

    mask = np.where((uff_pm.img == 0) & (rdf_pm.img == 1), 2, uff_pm.img)

    pmmoto.io.output.save_img(
        "examples/md_porous_media/image",
        subdomain,
        rdf_pm.img,
        additional_img={"uff": uff_pm.img, "mask": mask},
    )


def radii_plot(pmf_value):

    columbic_water = 1.4
    atom_folder = "examples/md_porous_media/rdf_bins/"
    atom_map = pmmoto.io.data_read.read_atom_map(atom_folder + "atom_map.txt")
    names = [entry["name"].replace("_", "-") for entry in atom_map.values()]

    pmf_radii = determine_pmf_radii(atom_folder, pmf_value)
    uff_radii = determine_uff_radii(atom_folder, columbic_water)

    labels = list(pmf_radii.keys())
    rdf_values = [pmf_radii[k] for k in labels]
    uff_values = [uff_radii[k] for k in labels]

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, rdf_values, width, label="Non-equilibrium")
    ax.bar(x + width / 2, uff_values, width, label="Equilibrium")

    # Labels and formatting
    ax.set_ylabel("Radius (Å)", fontsize=14)
    ax.set_xticks(x)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    fig.tight_layout()
    plt.savefig("examples/md_porous_media/radii_comparison.png")


if __name__ == "__main__":

    voxels_in = (800, 800, 800)
    membrane_file = "examples/md_porous_media/example_membrane_data.in"

    sd = initialize_domain(voxels_in)

    pmf = 17

    if rank == 0:
        radii_plot(pmf)

    # Plot for comparing radii from RDF and UFF
    # compare_radii(pmf, sd, membrane_file)
