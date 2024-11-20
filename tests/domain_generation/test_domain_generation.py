"""test_rdf.py"""

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

    sphere = np.array([0.9, 0.5, 0.5, 0.25])
    boundary_features = pmmoto.domain_generation.collect_boundary_crossings(
        sphere, padded_subdomain.domain.box
    )

    periodic_spheres = pmmoto.domain_generation.reflect_boundary_sphere(
        sphere, boundary_features, padded_subdomain
    )
