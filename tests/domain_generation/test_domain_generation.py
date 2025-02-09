"""test_rdf.py"""

import pytest
import numpy as np
import pmmoto


def test_pm_sphere():
    """
    Test domain generation of a sphere pack.
    Sphere data is [x_i, y_i, z_i, r_i]
    where x,y,z is the center of the sphere and r is the radius
    """
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    z = np.linspace(0, 1, 10)
    sphere = np.array([[0.5, 0.5, 0.5, 0.25]])

    grid = pmmoto.domain_generation._domain_generation.gen_pm_sphere(x, x, x, sphere)
    assert np.sum(grid) == 944


def test_pm_sphere_verlet():
    """
    Test domain generation of a sphere pack using Velet domains for speed-up.
    Sphere data is [x_i, y_i, z_i, r_i]
    where x,y,z is the center of the sphere and r is the radius
    """
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    z = np.linspace(0, 1, 10)

    sphere = np.array([[0.5, 0.5, 0.5, 0.25]])

    verlet = [3, 3, 3]

    grid = pmmoto.domain_generation._domain_generation.gen_pm_verlet_sphere(
        verlet, x, y, z, sphere
    )
    assert np.sum(grid) == 944


def test_pm_atom():
    """
    Test domain generation of a atomistic domain
    Atom locations is a NumPy array with centroid of each atom.
    Atom Types is a list of the type of each atom corresponding to the locations.
    Atom Cutoff is map that contains the cutoff distance for each atom type.

    """
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    z = np.linspace(0, 1, 10)

    atom_locations = np.array([[0.5, 0.5, 0.5]])

    atom_types = np.array([0, 0], dtype=np.int64)
    atom_cutoff = {}
    atom_cutoff[0] = 0.25

    spheres = pmmoto.domain_generation._domain_generation.convert_atoms_to_spheres(
        atom_locations, atom_types, atom_cutoff
    )

    np.testing.assert_array_equal(spheres, [[0.5, 0.5, 0.5, 0.25]])

    grid = pmmoto.domain_generation._domain_generation.gen_pm_atom(
        x, y, z, atom_locations, atom_types, atom_cutoff
    )
    assert np.sum(grid) == 944


def test_pm_atom_verlet():
    """
    Test domain generation of a atomistic domain
    Atom locations is a NumPy array with centroid of each atom.
    Atom Types is a list of the type of each atom corresponding to the locations.
    Atom Cutoff is map that contains the cutoff distance for each atom type.

    """
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    z = np.linspace(0, 1, 10)

    verlet = [1, 1, 1]

    atom_locations = np.array([[0.5, 0.5, 0.5]])

    atom_types = np.array([0], dtype=np.int64)
    atom_cutoff = {}
    atom_cutoff[0] = 0.25

    spheres = pmmoto.domain_generation._domain_generation.convert_atoms_to_spheres(
        atom_locations, atom_types, atom_cutoff
    )

    np.testing.assert_array_equal(spheres, [[0.5, 0.5, 0.5, 0.25]])

    grid = pmmoto.domain_generation._domain_generation.gen_pm_verlet_atom(
        verlet, x, y, z, atom_locations, atom_types, atom_cutoff
    )
    assert np.sum(grid) == 944


def test_is_inside_domain():
    """
    Test for the addition of spheres that are periodic and cross a boundary
    """
    domain_box = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    sphere = np.array([0.5, 0.5, 0.5, 0.25])

    assert pmmoto.domain_generation.is_inside_domain(sphere[0:3], domain_box)

    sphere = np.array([1.01, 0.5, 0.5, 0.25])

    assert not pmmoto.domain_generation.is_inside_domain(sphere[0:3], domain_box)


def test_collect_boundary_crossings():
    """
    Test for the addition of spheres that are periodic and cross a boundary
    """

    domain_box = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    sphere = np.array([0.5, 0.5, 0.5, 0.25])
    boundary_features = pmmoto.domain_generation.collect_boundary_crossings(
        sphere, domain_box
    )

    np.testing.assert_array_equal(boundary_features, [])

    sphere = np.array([0.9, 0.5, 0.1, 0.25])
    boundary_features = pmmoto.domain_generation.collect_boundary_crossings(
        sphere, domain_box
    )

    np.testing.assert_array_equal(
        boundary_features, [(1, 0, 0), (0, 0, -1), (1, 0, -1)]
    )

    sphere = np.array([0.1, 0.1, 0.1, 0.25])
    boundary_features = pmmoto.domain_generation.collect_boundary_crossings(
        sphere, domain_box
    )

    np.testing.assert_array_equal(
        boundary_features,
        [
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, -1),
            (-1, 0, -1),
            (-1, -1, 0),
            (0, -1, -1),
            (-1, -1, -1),
        ],
    )


def test_reflect_boundary_sphere(padded_subdomain):
    """
    Test addition of periodic spheres
    """

    # Face sphere
    periodic_features = pmmoto.core.subdomain_features.collect_periodic_features(
        padded_subdomain.features
    )

    periodic_corrections = pmmoto.core.subdomain_features.collect_periodic_corrections(
        padded_subdomain.features
    )
    sphere = np.array([[0.9, 0.5, 0.5, 0.25]])
    boundary_features = pmmoto.domain_generation.collect_boundary_crossings(
        sphere[0], padded_subdomain.domain.box
    )

    periodic_spheres = pmmoto.domain_generation.reflect_boundary_sphere(
        sphere[0],
        boundary_features,
        padded_subdomain.domain.length,
        periodic_features,
        periodic_corrections,
    )

    np.testing.assert_allclose(periodic_spheres, [[-0.1, 0.5, 0.5, 0.25]])

    # Edge sphere

    sphere = np.array([[0.1, 0.5, 0.1, 0.15]])
    boundary_features = pmmoto.domain_generation.collect_boundary_crossings(
        sphere[0], padded_subdomain.domain.box
    )

    periodic_spheres = pmmoto.domain_generation.reflect_boundary_sphere(
        sphere[0],
        boundary_features,
        padded_subdomain.domain.length,
        periodic_features,
        periodic_corrections,
    )

    np.testing.assert_allclose(
        periodic_spheres,
        [[1.1, 0.5, 0.1, 0.15], [0.1, 0.5, 1.1, 0.15], [0.1, 0.5, 0.1, 0.15]],
    )

    # Corner sphere

    sphere = np.array([[0.1, 0.1, 0.1, 0.45]])
    boundary_features = pmmoto.domain_generation.collect_boundary_crossings(
        sphere[0], padded_subdomain.domain.box
    )

    periodic_spheres = pmmoto.domain_generation.reflect_boundary_sphere(
        sphere[0],
        boundary_features,
        padded_subdomain.domain.length,
        periodic_features,
        periodic_corrections,
    )

    np.testing.assert_allclose(
        periodic_spheres,
        [
            [1.1, 0.1, 0.1, 0.45],
            [0.1, 1.1, 0.1, 0.45],
            [0.1, 0.1, 1.1, 0.45],
            [0.1, 0.1, 0.1, 0.45],
            [0.1, 0.1, 0.1, 0.45],
            [0.1, 0.1, 0.1, 0.45],
            [1.1, 1.1, 1.1, 0.45],
        ],
    )


def test_gen_periodic_spheres(padded_subdomain):
    """
    Test the addition of periodic spheres
    """

    # No periodic spheres

    spheres = np.array([[0.5, 0.5, 0.5, 0.25]])

    periodic_spheres = pmmoto.domain_generation.gen_periodic_spheres(
        padded_subdomain, spheres
    )

    np.testing.assert_allclose(periodic_spheres, [[0.5, 0.5, 0.5, 0.25]])

    spheres = np.array(
        [[0.9, 0.5, 0.5, 0.25], [0.1, 0.5, 0.1, 0.15], [0.1, 0.1, 0.1, 0.45]]
    )

    periodic_spheres = pmmoto.domain_generation.gen_periodic_spheres(
        padded_subdomain, spheres
    )

    np.testing.assert_allclose(
        periodic_spheres,
        [
            [0.9, 0.5, 0.5, 0.25],
            [0.1, 0.5, 0.1, 0.15],
            [0.1, 0.1, 0.1, 0.45],
            [-0.1, 0.5, 0.5, 0.25],
            [1.1, 0.5, 0.1, 0.15],
            [0.1, 0.5, 1.1, 0.15],
            [0.1, 0.5, 0.1, 0.15],
            [1.1, 0.1, 0.1, 0.45],
            [0.1, 1.1, 0.1, 0.45],
            [0.1, 0.1, 1.1, 0.45],
            [0.1, 0.1, 0.1, 0.45],
            [0.1, 0.1, 0.1, 0.45],
            [0.1, 0.1, 0.1, 0.45],
            [1.1, 1.1, 1.1, 0.45],
        ],
    )


def test_gen_periodic_atoms(padded_subdomain):
    """
    Test the addition of periodic spheres
    """

    # No periodic spheres

    atom_locations = np.array([[0.5, 0.5, 0.5]])

    atom_types = np.array([0], dtype=np.int64)
    atom_cutoff = {}
    atom_cutoff[0] = 0.25

    atom_locations, atom_types = pmmoto.domain_generation.gen_periodic_atoms(
        padded_subdomain, atom_locations, atom_types, atom_cutoff
    )

    np.testing.assert_allclose(atom_locations, [[0.5, 0.5, 0.5]])
    np.testing.assert_allclose(atom_types, [0])

    atom_locations = np.array([[0.1, 0.5, 0.5], [0.1, 0.5, 0.9], [0.97, 0.97, 0.97]])

    atom_types = np.array([0, 1, 12], dtype=np.int64)
    atom_cutoff = {}
    atom_cutoff[0] = 0.25
    atom_cutoff[1] = 0.35
    atom_cutoff[12] = 0.05

    atom_locations, atom_types = pmmoto.domain_generation.gen_periodic_atoms(
        padded_subdomain, atom_locations, atom_types, atom_cutoff
    )

    np.testing.assert_allclose(
        atom_locations,
        [
            [0.1, 0.5, 0.5],
            [0.1, 0.5, 0.9],
            [0.97, 0.97, 0.97],
            [1.1, 0.5, 0.5],
            [1.1, 0.5, 0.9],
            [0.1, 0.5, -0.1],
            [0.1, 0.5, 0.9],
            [-0.03, 0.97, 0.97],
            [0.97, -0.03, 0.97],
            [0.97, 0.97, -0.03],
            [0.97, 0.97, 0.97],
            [0.97, 0.97, 0.97],
            [0.97, 0.97, 0.97],
            [-0.03, -0.03, -0.03],
            [0.1, 0.5, 0.5],
            [0.1, 0.5, 0.9],
            [0.97, 0.97, 0.97],
        ],
    )

    np.testing.assert_allclose(
        atom_types, [0, 1, 12, 0, 0, 0, 0, 1, 1, 1, 1, 12, 12, 12, 12]
    )


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
