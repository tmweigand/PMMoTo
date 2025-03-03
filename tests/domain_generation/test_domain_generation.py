"""test_domain_generation.py"""

import pytest
import numpy as np
import pmmoto


def test_pm_sphere():
    """
    Test domain generation of a sphere pack.
    Sphere data is [x_i, y_i, z_i, r_i]
    where x,y,z is the center of the sphere and r is the radius
    """
    sd = pmmoto.initialize(voxels=(10, 10, 10))
    sphere = np.array([[0.5, 0.5, 0.5, 0.25]])

    img = pmmoto.domain_generation.gen_pm_sphere(sd, sphere)
    assert np.sum(img) == 944


def test_pm_sphere_verlet():
    """
    Test domain generation of a sphere pack using Velet domains for speed-up.
    Sphere data is [x_i, y_i, z_i, r_i]
    where x,y,z is the center of the sphere and r is the radius
    """
    sd = pmmoto.initialize(voxels=(10, 10, 10), verlet_domains=(3, 3, 3))
    sphere = np.array([[0.5, 0.5, 0.5, 0.25]])
    img = pmmoto.domain_generation.gen_pm_sphere(sd, sphere)

    assert np.sum(img) == 944


def test_pm_atom():
    """
    Test domain generation of a atomistic domain
    Atom locations is a NumPy array with centroid of each atom.
    Atom Types is a list of the type of each atom corresponding to the locations.
    Atom Cutoff is map that contains the cutoff distance for each atom type.

    """
    sd = pmmoto.initialize(voxels=(10, 10, 10))
    atom_locations = np.array([[0.5, 0.5, 0.5]])

    atom_types = np.array([0, 0], dtype=np.int64)
    atom_cutoff = {}
    atom_cutoff[0] = 0.25

    spheres = pmmoto.domain_generation._domain_generation.convert_atoms_to_spheres(
        atom_locations, atom_types, atom_cutoff
    )

    np.testing.assert_array_equal(spheres, [[0.5, 0.5, 0.5, 0.25]])

    grid = pmmoto.domain_generation._domain_generation.gen_pm_atom(
        sd, atom_locations, atom_types, atom_cutoff
    )
    assert np.sum(grid) == 944


def test_pm_atom_verlet():
    """
    Test domain generation of a atomistic domain
    Atom locations is a NumPy array with centroid of each atom.
    Atom Types is a list of the type of each atom corresponding to the locations.
    Atom Cutoff is map that contains the cutoff distance for each atom type.

    """
    sd = pmmoto.initialize(voxels=(10, 10, 10))

    atom_locations = np.array([[0.5, 0.5, 0.5]])

    atom_types = np.array([0], dtype=np.int64)
    atom_cutoff = {}
    atom_cutoff[0] = 0.25

    spheres = pmmoto.domain_generation._domain_generation.convert_atoms_to_spheres(
        atom_locations, atom_types, atom_cutoff
    )

    np.testing.assert_array_equal(spheres, [[0.5, 0.5, 0.5, 0.25]])

    img = pmmoto.domain_generation._domain_generation.gen_pm_atom(
        sd, atom_locations, atom_types, atom_cutoff
    )

    assert np.sum(img) == 944


@pytest.mark.figures
def test_gen_random_binary_grid():
    """
    Test domain generation of a random binary grid
    """
    voxels = (50, 50, 50)

    img = pmmoto.domain_generation.domain_generation.gen_random_binary_grid(
        voxels, p_zero=0.2, seed=1
    )
    pmmoto.io.output.save_img("data_out/test_random_binary_grid", img)


@pytest.mark.figures
def test_gen_smoothed_random_binary_grid():
    """
    Test domain generation of a random binary grid
    """
    voxels = (100, 100, 100)

    img = pmmoto.domain_generation.domain_generation.gen_smoothed_random_binary_grid(
        voxels, p_zero=0.5, smoothness=2.0, seed=1
    )

    pmmoto.io.output.save_img("data_out/test_smoothed_random_binary_grid", img)
