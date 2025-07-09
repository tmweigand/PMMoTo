"""test_domain_generation.py"""

import pytest
import numpy as np
import pmmoto


def test_pm_sphere() -> None:
    """Test domain generation of a sphere pack.

    Sphere data is [x_i, y_i, z_i, r_i]
    where x,y,z is the center of the sphere and r is the radius
    """
    sd = pmmoto.initialize(voxels=(10, 10, 10))
    sphere = np.array([[0.5, 0.5, 0.5, 0.25]])

    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, sphere)
    assert np.sum(pm.img) == 944


def test_pm_sphere_verlet() -> None:
    """Test domain generation of a sphere pack using Velet domains for speed-up.

    Sphere data is [x_i, y_i, z_i, r_i]
    where x,y,z is the center of the sphere and r is the radius
    """
    sd = pmmoto.initialize(voxels=(10, 10, 10), verlet_domains=(3, 3, 3))
    sphere = np.array([[0.5, 0.5, 0.5, 0.25]])
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, sphere)

    assert np.sum(pm.img) == 944


def test_pm_atom() -> None:
    """Test domain generation of a atomistic domain.

    Atom locations is a NumPy array with centroid of each atom.
    Atom Types is a list of the type of each atom corresponding to the locations.
    Atom Cutoff is map that contains the cutoff distance for each atom type.

    """
    sd = pmmoto.initialize(voxels=(10, 10, 10))
    atom_locations = np.array([[0.5, 0.5, 0.5]])

    atom_types = np.array([1], dtype=np.int64)
    atom_radii = {}
    atom_radii[1] = 0.25

    pm = pmmoto.domain_generation.gen_pm_atom_domain(
        sd, atom_locations, atom_radii, atom_types
    )
    assert np.sum(pm.img) == 944


def test_pm_atom_verlet() -> None:
    """Test domain generation of a atomistic domain.

    Atom locations is a NumPy array with centroid of each atom.
    Atom Types is a list of the type of each atom corresponding to the locations.
    Atom Cutoff is map that contains the cutoff distance for each atom type.

    """
    sd = pmmoto.initialize(voxels=(10, 10, 10))

    atom_locations = np.array([[0.5, 0.5, 0.5]])

    atom_types = np.array([1], dtype=np.int64)
    atom_radii = {}
    atom_radii[1] = 0.25

    pm = pmmoto.domain_generation.gen_pm_atom_domain(
        sd, atom_locations, atom_radii, atom_types
    )

    assert np.sum(pm.img) == 944


@pytest.mark.figures
def test_gen_random_binary_grid() -> None:
    """Test domain generation of a random binary grid"""
    voxels = (50, 50, 50)

    img = pmmoto.domain_generation.domain_generation.gen_img_random_binary(
        voxels, p_zero=0.2, seed=1
    )
    # pmmoto.io.output.save_img("data_out/test_random_binary_grid", img)


@pytest.mark.figures
def test_gen_smoothed_random_binary_grid() -> None:
    """Test domain generation of a random binary grid"""
    voxels = (100, 100, 100)

    img = pmmoto.domain_generation.domain_generation.gen_img_smoothed_random_binary(
        voxels, p_zero=0.5, smoothness=2.0, seed=1
    )

    # pmmoto.io.output.save_img("data_out/test_smoothed_random_binary_grid", img)


def test_gen_cylinders() -> None:
    """Tests for cylinder pack"""
    sd = pmmoto.initialize(voxels=(100, 100, 100))
    cylinder = np.array(
        [
            [0.5, 0.5, -0.05, 0.5, 0.5, 1.05, 0.15],
            [0.25, 0.25, 0.55, 0.75, 0.75, 0.85, 0.15],
        ]
    )

    pm = pmmoto.domain_generation.gen_pm_cylinders(sd, cylinder)

    # pmmoto.io.output.save_img("data_out/cylinders", pm.img, sd.domain.resolution)

    # assert np.sum(pm.img) == 944


@pytest.mark.mpi(min_size=8)
def test_deconstruct_img():
    """Ensure expected behavior of deconstruct_grid"""
    boundary_types = (
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
    )
    sd = pmmoto.initialize(voxels=(100, 100, 100), boundary_types=boundary_types)

    n = sd.domain.voxels[0]
    linear_values = np.linspace(0, n - 1, n, endpoint=True)
    img = np.ones(sd.domain.voxels) * linear_values

    subdomains, local_img = pmmoto.domain_generation.deconstruct_img(
        sd, img, subdomains=(2, 2, 2)
    )

    subdomains, local_img = pmmoto.domain_generation.deconstruct_img(
        sd, img, subdomains=(2, 2, 2), rank=2
    )
