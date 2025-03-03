"""test_particles.py"""

import numpy as np
from mpi4py import MPI
import pmmoto


def test_particles():
    """
    Test for generating a radial distribution function form atom data
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    N = 50
    atoms = np.random.rand(N, 4)
    atoms[:, 3] = atoms[:, 3]

    eps = 0
    box = ((eps, 1 - eps), (eps, 1 - eps), (eps, 1 - eps))
    subdomains = (2, 1, 1)

    sd = pmmoto.initialize(
        voxels=(10, 10, 10),
        boundary_types=((2, 2), (2, 2), (2, 2)),
        rank=rank,
        subdomains=subdomains,
    )

    atoms = pmmoto.domain_generation.particles.initialize(sd, atoms)

    pmmoto.io.output.save_img_data_parallel(
        "data_out/test_particles_subdomain", sd, np.zeros(sd.voxels)
    )
    pmmoto.io.output.save_particle_data("data_out/test_particles", sd, atoms)


def test_gen_periodic_spheres():
    """
    Test the addition of periodic spheres
    """

    sd = pmmoto.initialize(voxels=(10, 10, 10), boundary_types=((2, 2), (2, 2), (2, 2)))

    # No periodic spheres
    spheres = np.array([[0.5, 0.5, 0.5, 0.25]])
    sphere_list = pmmoto.domain_generation.particles.initialize(sd, spheres)

    np.testing.assert_allclose(sphere_list.return_np_array(), [[0.5, 0.5, 0.5, 0.25]])

    spheres = np.array(
        [[0.9, 0.5, 0.5, 0.25], [0.1, 0.5, 0.1, 0.15], [0.1, 0.1, 0.1, 0.45]]
    )

    sphere_list = pmmoto.domain_generation.particles.initialize(
        sd, spheres, add_periodic=True
    )

    np.testing.assert_allclose(
        sphere_list.return_np_array(),
        [
            [-0.1, 0.5, 0.5, 0.25],
            [0.9, 0.5, 0.5, 0.25],
            [0.1, 0.5, 0.1, 0.15],
            [0.1, 0.5, 1.1, 0.15],
            [1.1, 0.5, 0.1, 0.15],
            [1.1, 0.5, 1.1, 0.15],
            [0.1, 0.1, 0.1, 0.45],
            [0.1, 0.1, 1.1, 0.45],
            [0.1, 1.1, 0.1, 0.45],
            [0.1, 1.1, 1.1, 0.45],
            [1.1, 0.1, 0.1, 0.45],
            [1.1, 0.1, 1.1, 0.45],
            [1.1, 1.1, 0.1, 0.45],
            [1.1, 1.1, 1.1, 0.45],
        ],
    )


def test_trim_particles():
    """
    Test the addition of periodic spheres
    """

    sd = pmmoto.initialize(voxels=(10, 10, 10))

    particles = np.array([[0.5, 0.5, 0.5, 0.25], [1.1, 0.5, 0.5, 0.09]])

    trimmed_particles = pmmoto.domain_generation.particles.initialize(sd, particles)

    np.testing.assert_allclose(
        trimmed_particles.return_np_array(return_own=True),
        np.array([[0.5, 0.5, 0.5, 0.25, 1]]),
    )

    particles = np.array([[0.5, 0.5, 0.5, 0.25], [1.08, 0.5, 0.5, 0.09]])

    trimmed_particles = pmmoto.domain_generation.particles.initialize(sd, particles)

    np.testing.assert_allclose(
        trimmed_particles.return_np_array(return_own=True),
        np.array([[0.5, 0.5, 0.5, 0.25, 1], [1.08, 0.5, 0.5, 0.09, 0]]),
    )
