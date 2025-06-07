"""Unit tests for PMMoTo Minkowski functionals"""

import numpy as np
from mpi4py import MPI
import pmmoto
import pytest


def test_minkowski() -> None:
    """Ensure single sphere is correct.

    Note: The curvature and Euler characteristic is poor at this resolution
    """
    voxels = (100, 100, 100)

    file_in = "./tests/test_domains/single_sphere.in"

    sd = pmmoto.initialize(voxels=voxels, verlet_domains=(1, 1, 1))
    sphere_data, domain_data = pmmoto.io.read_sphere_pack_xyzr_domain(file_in)
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, sphere_data, domain_data)

    mink_funcs = pmmoto.analysis.minkowski.functionals(sd, np.logical_not(pm.img))

    radius = sphere_data[0, 3]

    vol = 4 / 3 * np.pi * radius**3
    sa = 4 * np.pi * radius * radius
    curv = sa * ((1.0 / radius) + (1.0 / radius))
    euler = sa / (radius * radius) / (4 * np.pi)

    np.testing.assert_almost_equal(mink_funcs[0], vol, decimal=1)
    np.testing.assert_almost_equal(mink_funcs[1], sa, decimal=1)
    np.testing.assert_almost_equal(mink_funcs[2], 8.8, decimal=1)
    np.testing.assert_almost_equal(mink_funcs[3], -2.0, decimal=1)


@pytest.mark.mpi(min_size=8)
def test_minkowski_functionals() -> None:
    """Ensure correct parallel implementation"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    voxels = (100, 100, 100)
    subdomains = (2, 2, 2)
    boundary_value = [0, 1, 2]

    for b_value in boundary_value:
        boundary_types = ((b_value, b_value), (b_value, b_value), (b_value, b_value))

        file_in = "./tests/test_domains/single_sphere.in"
        sphere_data, domain_data = pmmoto.io.read_sphere_pack_xyzr_domain(file_in)

        sd_single = pmmoto.initialize(voxels=voxels, boundary_types=boundary_types)
        pm_single = pmmoto.domain_generation.gen_pm_spheres_domain(
            sd_single, sphere_data, domain_data
        )
        mink_funcs_single = pmmoto.analysis.minkowski.functionals(
            sd_single, np.logical_not(pm_single.img)
        )

        sd_parallel = pmmoto.initialize(voxels=voxels, subdomains=subdomains, rank=rank)
        pm_parallel = pmmoto.domain_generation.gen_pm_spheres_domain(
            sd_parallel, sphere_data, domain_data
        )
        mink_funcs_parallel = pmmoto.analysis.minkowski.functionals(
            sd_parallel, np.logical_not(pm_parallel.img)
        )

        np.testing.assert_equal(mink_funcs_single[0], mink_funcs_parallel[0])
        np.testing.assert_equal(mink_funcs_single[1], mink_funcs_parallel[1])
        np.testing.assert_equal(mink_funcs_single[2], mink_funcs_parallel[2])
        np.testing.assert_equal(mink_funcs_single[3], mink_funcs_parallel[3])
