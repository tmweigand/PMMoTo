"""test_domain_generation.py"""

import pmmoto
import pytest
import numpy as np


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


def test_gen_random_binary_grid() -> None:
    """Test domain generation of a random binary grid"""
    voxels = (50, 50, 50)

    img = pmmoto.domain_generation.domain_generation.gen_img_random_binary(
        voxels, p_zero=0.2, seed=1
    )

    assert img.shape == (50, 50, 50)

    with pytest.raises(ValueError):
        _ = pmmoto.domain_generation.domain_generation.gen_img_random_binary(
            voxels, p_zero=1.5, seed=1
        )


def test_gen_smoothed_random_binary_grid() -> None:
    """Test domain generation of a random binary grid"""
    voxels = (100, 100, 100)
    img = pmmoto.domain_generation.domain_generation.gen_img_smoothed_random_binary(
        voxels, p_zero=0.5, smoothness=2.0, seed=1
    )
    assert img.shape == (100, 100, 100)

    with pytest.raises(ValueError):
        _ = pmmoto.domain_generation.domain_generation.gen_img_smoothed_random_binary(
            voxels, p_zero=1.5, smoothness=2.0, seed=1
        )

    with pytest.raises(ValueError):
        _ = pmmoto.domain_generation.domain_generation.gen_img_smoothed_random_binary(
            voxels, p_zero=0.5, smoothness=-5, seed=1
        )


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
    assert np.sum(pm.img) == 893668


def test_gen_pm_atom_file(tmp_path):
    """Test generation of img from atom file"""
    dummy_file = tmp_path / "dummy_lammps.data"
    dummy_file.write_text(
        """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id mol type mass q x y z v_peratompress c_peratomvol[1]
1 1 1 12.01 0.0 1.0 1.0 1.0 0.0 0.0
2 1 2 16.0 0.0 2.0 2.0 2.0 0.0 0.0
    """
    )
    sd = pmmoto.initialize((10, 10, 10), box=((0, 10), (0, 10), (0, 10)))

    atom_radii = {1: 2, 2: 4}
    pm = pmmoto.domain_generation.gen_pm_atom_file(
        sd, str(dummy_file), atom_radii=atom_radii
    )

    assert pm.porosity == pytest.approx(0.84)


def test_deconstruct_img():
    """Ensure expected behavior of deconstruct_grid"""
    boundary_types = (
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
    )
    subdomains = (2, 1, 1)
    sd = pmmoto.initialize(
        voxels=(4, 4, 4), boundary_types=boundary_types, subdomains=subdomains
    )

    n = sd.domain.voxels[0]
    linear_values = np.linspace(0, n - 1, n, endpoint=True)
    img = np.ones(sd.domain.voxels) * linear_values

    subdomains, local_img = pmmoto.domain_generation.deconstruct_img(
        sd, img, subdomains=(2, 1, 1)
    )

    keys = [0, 1]
    shape = (4, 6, 6)
    pattern = np.array([3.0, 0.0, 1.0, 2.0, 3.0, 0.0])

    # Vectorized dictionary generation
    expected_arrays = {k: np.tile(pattern, (shape[0], shape[1], 1)) for k in keys}

    np.testing.assert_array_equal(local_img[0], expected_arrays[0])
    np.testing.assert_array_equal(local_img[1], expected_arrays[1])

    with pytest.raises(ValueError):
        img = np.zeros([2, 2, 2])
        _ = pmmoto.domain_generation.deconstruct_img(sd, img, subdomains=(2, 1, 1))
