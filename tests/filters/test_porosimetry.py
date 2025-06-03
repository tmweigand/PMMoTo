"""test_porosimetry.py"""

import pytest
import pmmoto
import numpy as np
from mpi4py import MPI


def test_porosimetry_sizes():
    """Test probe diamterr sizes for linear and logarithmic"""
    min_value = 0
    max_value = 10
    num_values = 4
    values = pmmoto.filters.porosimetry.get_sizes(
        min_value, max_value, num_values, "linear"
    )

    # linear test
    np.testing.assert_array_almost_equal(values, [10.0, 6.666667, 3.333333, 0.0])

    min_log_value = 1
    values = pmmoto.filters.porosimetry.get_sizes(
        min_log_value, max_value, num_values, "log"
    )
    # print(values)
    # log test
    np.testing.assert_array_almost_equal(values, [10.0, 4.64158883, 2.15443469, 1.0])


@pytest.mark.xfail
def test_porosimetry_sizes_input_fail():
    """Ensuring checks are behaving correctly"""
    _ = pmmoto.filters.porosimetry.get_sizes(5, 1, 5)
    _ = pmmoto.filters.porosimetry.get_sizes(0, 10, 0)


def test_modes():
    """Test morph, dt, and hybrid porosimetry modes.

    Expected output is all should be equal to eachother.
    """
    radius = 0.030001
    spheres, domain_data = pmmoto.io.data_read.read_sphere_pack_xyzr_domain(
        "tests/test_domains/bcc.in"
    )
    sd = pmmoto.initialize(
        rank=0,
        box=domain_data,
        boundary_types=((0, 0), (0, 0), (0, 0)),
        voxels=(100, 100, 100),
    )

    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres)
    morph = pmmoto.filters.porosimetry.porosimetry(sd, pm, radius, "morph")
    dt_mode = pmmoto.filters.porosimetry.porosimetry(sd, pm, radius, "distance")
    hybrid = pmmoto.filters.porosimetry.porosimetry(sd, pm, radius, "hybrid")

    pmmoto.io.output.save_img_data_parallel(
        "data_out/test_modes",
        sd,
        pm.img,
        additional_img={
            "morph": morph,
            "dt_mode": dt_mode,
            "hybrid": hybrid,
        },
    )

    np.testing.assert_array_almost_equal(morph, dt_mode)
    np.testing.assert_array_almost_equal(morph, hybrid)


def test_porosimetry_inlet():
    """Ensure when inlet = True, that only voxels connected in inlet are > 0"""
    radius = 0.02
    voxels = (100, 100, 100)
    inlet = ((1, 0), (0, 0), (0, 0))
    sd = pmmoto.initialize(voxels=voxels, inlet=inlet)

    img = np.zeros(sd.voxels)
    img[0:10, 5:10, 5:10] = 1
    img[12:35, 12:35, 12:35] = 1

    pm = pmmoto.domain_generation.porousmedia.gen_pm(subdomain=sd, img=img)

    morph_no_inlet = pmmoto.filters.porosimetry.porosimetry(
        subdomain=sd, porous_media=pm, radius=radius, mode="morph", inlet=False
    )

    morph_inlet = pmmoto.filters.porosimetry.porosimetry(
        subdomain=sd, porous_media=pm, radius=radius, mode="morph", inlet=True
    )

    pmmoto.io.output.save_img_data_parallel(
        "data_out/test_morph_inlet",
        sd,
        img,
        additional_img={
            "morph_no_inlet": morph_no_inlet,
            "morph_inlet": morph_inlet,
        },
    )


def test_pore_size_distribution():
    """Test generation of pore size distribution for an inkbottle"""
    voxels = (560, 120, 120)
    box = ((0.0, 14.0), (-1.5, 1.5), (-1.5, 1.5))
    inlet = ((0, 1), (0, 0), (0, 0))
    sd = pmmoto.initialize(voxels, box, inlet=inlet)
    pm = pmmoto.domain_generation.gen_pm_inkbottle(sd)
    img = pmmoto.filters.porosimetry.pore_size_distribution(sd, pm, inlet=True)

    pmmoto.io.output.save_img_data_parallel(
        "data_out/inkbottle_ps_distribution",
        sd,
        pm.img,
        additional_img={"psd": img, "edt": pm.distance},
    )


@pytest.mark.skip
def test_lammps_psd():
    """Test for reading membrane files for psd."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    lammps_file = "tests/test_data/LAMMPS/membranedata.gz"

    positions, types, box, time = pmmoto.io.data_read.read_lammps_atoms(lammps_file)

    unique_types = np.unique(types)
    radii = {}
    for _id in unique_types:
        radii[_id] = 2

    # ignore reservoirs
    box[2] = [-100, 100]

    sd = pmmoto.initialize(
        voxels=(500, 500, 500),
        box=box,
        rank=rank,
        subdomains=(2, 2, 2),
        boundary_types=((2, 2), (2, 2), (2, 2)),
        verlet_domains=[20, 20, 20],
    )

    pm = pmmoto.domain_generation.gen_pm_atom_domain(sd, positions, radii, types)
    img = pmmoto.filters.porosimetry.pore_size_distribution(sd, pm, inlet=False)

    pmmoto.io.output.save_img_data_parallel(
        "data_out/membrane_binary_map",
        sd,
        pm.img,
        additional_img={"psd": img},
    )
