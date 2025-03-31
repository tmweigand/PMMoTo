"""test_particles.py"""

import numpy as np
from mpi4py import MPI
import pmmoto
import gc


def test_particles():
    """
    Test for generating a radial distribution function form atom data
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    N = 50
    spheres = np.random.rand(N, 4)

    eps = 0
    box = ((eps, 1 - eps), (eps, 1 - eps), (eps, 1 - eps))
    subdomains = (2, 1, 1)

    sd = pmmoto.initialize(
        voxels=(10, 10, 10),
        boundary_types=((2, 2), (2, 2), (2, 2)),
        box=box,
        rank=rank,
        subdomains=subdomains,
    )

    spheres = pmmoto.particles.initialize_spheres(sd, spheres)

    pmmoto.io.output.save_img_data_parallel(
        "data_out/test_particles_subdomain", sd, np.zeros(sd.voxels)
    )
    pmmoto.io.output.save_particle_data(
        "data_out/test_particles", sd, spheres.return_np_array()
    )


def test_gen_periodic_spheres():
    """
    Test the addition of periodic spheres
    """

    sd = pmmoto.initialize(voxels=(10, 10, 10), boundary_types=((2, 2), (2, 2), (2, 2)))

    # No periodic spheres
    spheres = np.array([[0.5, 0.5, 0.5, 0.25]])
    sphere_list = pmmoto.particles.initialize_spheres(sd, spheres)

    np.testing.assert_allclose(sphere_list.return_np_array(), [[0.5, 0.5, 0.5, 0.25]])

    spheres = np.array(
        [[0.9, 0.5, 0.5, 0.25], [0.1, 0.5, 0.1, 0.15], [0.1, 0.1, 0.1, 0.45]]
    )

    sphere_list = pmmoto.particles.initialize_spheres(sd, spheres, add_periodic=True)

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

    spheres = np.array([[0.5, 0.5, 0.5, 0.25], [1.1, 0.5, 0.5, 0.09]])

    trimmed_spheres = pmmoto.particles.initialize_spheres(
        sd, spheres, trim_intersecting=True, set_own=True
    )

    np.testing.assert_allclose(
        trimmed_spheres.return_np_array(return_own=True),
        np.array([[0.5, 0.5, 0.5, 0.25, 1]]),
    )

    spheres = np.array([[0.5, 0.5, 0.5, 0.25], [1.08, 0.5, 0.5, 0.09]])

    trimmed_spheres = pmmoto.particles.initialize_spheres(sd, spheres)

    np.testing.assert_allclose(
        trimmed_spheres.return_np_array(return_own=True),
        np.array([[0.5, 0.5, 0.5, 0.25, 1], [1.08, 0.5, 0.5, 0.09, 0]]),
    )


def test_group_atoms():
    """
    Test the creation of atom lists
    """
    sd = pmmoto.initialize(voxels=(10, 10, 10), boundary_types=((2, 2), (2, 2), (2, 2)))

    # No periodic spheres
    atom_coordinates = np.array(
        [
            [0.05, 0.5, 0.5],
            [0.05, 0.5, 0.5],
            [0.05, 0.5, 0.5],
            [0.05, 0.5, 0.5],
            [25, 0.5, 0.5],
        ]
    )
    atom_ids = np.array([1, 15, 3, 15, 15], dtype=int)

    atom_radii = {}
    for _id in atom_ids:
        atom_radii[_id] = 0.1

    atoms = pmmoto.particles.initialize_atoms(
        sd, atom_coordinates, atom_radii, atom_ids, by_type=False
    )
    atoms.build_KDtree()
    atoms = atoms.return_np_array(True)


def test_spheres():
    """
    Test the creation of sphere lists
    """
    sd = pmmoto.initialize(voxels=(10, 10, 10), boundary_types=((2, 2), (2, 2), (2, 2)))

    # No periodic spheres
    sphere = np.array([[0.19, 0.1, 0.5, 0.2]])
    spheres = pmmoto.particles.initialize_spheres(sd, sphere, add_periodic=True)

    spheres.build_KDtree()

    print(spheres.return_np_array())


def test_cleanup():
    """
    Test deletion of particle lists
    """
    sd = pmmoto.initialize(voxels=(10, 10, 10), boundary_types=((2, 2), (2, 2), (2, 2)))

    # No periodic spheres
    atom_coordinates = np.array(
        [
            [0.05, 0.5, 0.5],
            [0.05, 0.5, 0.5],
            [0.05, 0.5, 0.5],
            [0.05, 0.5, 0.5],
            [25, 0.5, 0.5],
        ]
    )
    atom_ids = np.array([1, 15, 3, 15, 15], dtype=int)

    atom_radii = {_id: 0.1 for _id in atom_ids}

    atoms = pmmoto.particles.initialize_atoms(
        sd, atom_coordinates, atom_radii, atom_ids, by_type=False
    )


def test_uff_radius():
    """
    Test for the universal force field lookup to convert atoms to radii.
    """

    atom_names = ["C", "H", "N", "O"]
    radii_names = pmmoto.particles.uff_radius(atom_names=atom_names)

    assert radii_names == {6: 1.7155, 1: 1.2855, 7: 1.6305, 8: 1.559}

    atomic_numbers = [6, 1, 7, 8]
    radii_number = pmmoto.particles.uff_radius(atomic_numbers=atomic_numbers)

    assert radii_names == radii_number


def test_atoms_with_masses():
    """
    Test deletion of particle lists
    """
    sd = pmmoto.initialize(voxels=(10, 10, 10), boundary_types=((2, 2), (2, 2), (2, 2)))

    # No periodic spheres
    atom_coordinates = np.array(
        [
            [0.05, 0.5, 0.5],
            [0.05, 0.5, 0.5],
            [0.05, 0.5, 0.5],
            [0.05, 0.5, 0.5],
            [25, 0.5, 0.5],
        ]
    )
    atom_ids = np.array([1, 15, 3, 15, 15], dtype=int)

    atom_radii = {_id: 0.1 for _id in atom_ids}

    atom_masses = {_id: 0.3 for _id in atom_ids}

    atoms = pmmoto.particles.initialize_atoms(
        sd, atom_coordinates, atom_radii, atom_ids, atom_masses, by_type=False
    )

    print(atoms.return_masses())
