"""test_particles.py"""

import pmmoto
import pytest
import numpy as np
from mpi4py import MPI


def test_particles():
    """Test for generating a radial distribution function form atom data"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    N = 50
    spheres = np.random.rand(N, 4)

    eps = 0
    box = ((eps, 1 - eps), (eps, 1 - eps), (eps, 1 - eps))
    subdomains = (2, 1, 1)

    sd = pmmoto.initialize(
        voxels=(10, 10, 10),
        boundary_types=(
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        ),
        box=box,
        rank=rank,
        subdomains=subdomains,
    )

    spheres = pmmoto.particles.initialize_spheres(sd, spheres)

    pmmoto.io.output.save_img(
        "data_out/test_particles_subdomain", sd, np.zeros(sd.voxels)
    )
    pmmoto.io.output.save_particle_data(
        "data_out/test_particles", sd, spheres.return_np_array()
    )


def test_gen_periodic_spheres():
    """Test the addition of periodic spheres"""
    sd = pmmoto.initialize(
        voxels=(10, 10, 10),
        boundary_types=(
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        ),
    )

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
    """Test the addition of periodic spheres"""
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
    """Test the creation of atom lists"""
    sd = pmmoto.initialize(
        voxels=(10, 10, 10),
        boundary_types=(
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        ),
    )

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
    """Test the creation of sphere lists"""
    sd = pmmoto.initialize(
        voxels=(10, 10, 10),
        boundary_types=(
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        ),
    )

    # No periodic spheres
    sphere = np.array([[0.19, 0.1, 0.5, 0.2]])
    spheres = pmmoto.particles.initialize_spheres(sd, sphere, add_periodic=True)

    spheres.build_KDtree()


def test_cleanup():
    """Test deletion of particle lists"""
    sd = pmmoto.initialize(
        voxels=(10, 10, 10),
        boundary_types=(
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        ),
    )

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

    _ = pmmoto.particles.initialize_atoms(
        sd, atom_coordinates, atom_radii, atom_ids, by_type=False
    )


def test_uff_radius():
    """Test for the universal force field lookup to convert atoms to radii."""
    atom_names = ["C", "H", "N", "O"]
    radii_names = pmmoto.particles.uff_radius(atom_names=atom_names)

    assert radii_names == {6: 1.7155, 1: 1.2855, 7: 1.6305, 8: 1.559}

    atomic_numbers = [6, 1, 7, 8]
    radii_number = pmmoto.particles.uff_radius(atomic_numbers=atomic_numbers)

    assert radii_names == radii_number


def test_atoms_with_masses():
    """Test deletion of particle lists"""
    sd = pmmoto.initialize(
        voxels=(10, 10, 10),
        boundary_types=(
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        ),
    )

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

    _ = pmmoto.particles.initialize_atoms(
        sd, atom_coordinates, atom_radii, atom_ids, atom_masses, by_type=False
    )


def test_count_own():
    """Test deletion of particle lists"""
    sd = pmmoto.initialize(
        voxels=(10, 10, 10),
        boundary_types=(
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        ),
    )

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

    assert atoms.get_own_count() == 4
    assert atoms.size() == 5

    atoms = pmmoto.particles.initialize_atoms(
        sd, atom_coordinates, atom_radii, atom_ids, atom_masses, by_type=True
    )

    assert atoms.size() == {1: 1, 3: 1, 15: 3}
    assert atoms.get_own_count() == {1: 1, 3: 1, 15: 2}


def test_elements_to_ids():
    """Tests for conversion"""
    atom_elements = ["H", "C", "O", "N", "Cl"]

    atom_ids = pmmoto.particles.convert_atoms_elements_to_ids(atom_elements)

    np.testing.assert_array_equal(atom_ids, np.array([1, 6, 8, 7, 17]))


def test_load_uff_data_from_custom_file(tmp_path):
    """Test _load_uff_data with a custom UFF file"""
    # Create a custom UFF file
    uff_file = tmp_path / "custom_uff.txt"
    uff_content = """AtomicNumber Name Diameter
1 H 2.886
6 C 3.851
8 O 3.500
7 N 3.660
17 Cl
"""
    uff_file.write_text(uff_content)

    # Load data from custom file
    element_table = pmmoto.particles.particles._load_uff_data(str(uff_file))

    # Test lookup by name
    assert "H" in element_table
    assert "C" in element_table
    assert "O" in element_table
    assert "N" in element_table
    assert "Cl" not in element_table

    # Test lookup by atomic number
    assert 1 in element_table
    assert 6 in element_table
    assert 8 in element_table
    assert 7 in element_table

    # Test values (diameter should be halved to radius)
    atomic_num_h, radius_h = element_table["H"]
    assert atomic_num_h == 1
    assert radius_h == pytest.approx(2.886 / 2.0)

    atomic_num_c, radius_c = element_table[6]
    assert atomic_num_c == 6
    assert radius_c == pytest.approx(3.851 / 2.0)


def test_uff_radius_inputs():
    """Ensures correct behavior for inputs"""
    atom_names = ["C", "H"]
    atomic_numbers = [6, 1]

    with pytest.raises(ValueError):
        _ = pmmoto.particles.uff_radius(atom_names, atomic_numbers)


def test_atom_initialization_order():
    """Test initialization of atoms"""
    sd = pmmoto.initialize((10, 10, 10))
    atom_coordinates = np.array([[0.0, 0.0, 0.0], [0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    atom_coordinates_f = np.array(atom_coordinates, order="F")

    atom_radii = {1: 0.2, 2: 0.1, 3: 0.05}
    atom_ids = np.array([1, 2, 3])

    assert not atom_coordinates_f.flags["C_CONTIGUOUS"]

    particles_c = pmmoto.particles.initialize_atoms(
        sd, atom_coordinates, atom_radii, atom_ids
    )

    particles_f = pmmoto.particles.initialize_atoms(
        sd, atom_coordinates_f, atom_radii, atom_ids
    )

    np.testing.assert_array_equal(
        particles_c.return_np_array(), particles_f.return_np_array()
    )


def test_atom_initialization():
    """Test initialization of atoms"""
    sd = pmmoto.initialize((10, 10, 10))
    atom_coordinates = np.array(
        [[0.0, 0.0, 0.0], [0, 1.0, 1.0], [0.5, 0.5, 0.5], [1.1, 1.1, 1.1]]
    )

    atom_radii = {1: 0.2, 2: 0.1, 3: 0.05}
    atom_ids = np.array([1, 2, 3, 1])

    # Trim last atom
    particles = pmmoto.particles.initialize_atoms(
        sd, atom_coordinates, atom_radii, atom_ids, trim_within=True
    )

    np.testing.assert_array_equal(
        particles.return_np_array(),
        np.array([[0.0, 0.0, 0.0, 0.2], [0.0, 1.0, 1.0, 0.1], [0.5, 0.5, 0.5, 0.05]]),
    )

    # Trim intersecting so keep last atom as intersects boundary
    particles = pmmoto.particles.initialize_atoms(
        sd, atom_coordinates, atom_radii, atom_ids, trim_intersecting=True
    )

    np.testing.assert_array_equal(
        particles.return_np_array(),
        np.array(
            [
                [0.0, 0.0, 0.0, 0.2],
                [0.0, 1.0, 1.0, 0.1],
                [0.5, 0.5, 0.5, 0.05],
                [1.1, 1.1, 1.1, 0.2],
            ]
        ),
    )

    # Trim within and add periodic but no periodic boundaries!
    particles = pmmoto.particles.initialize_atoms(
        sd,
        atom_coordinates,
        atom_radii,
        atom_ids,
        trim_within=True,
        add_periodic=True,
    )

    np.testing.assert_array_equal(
        particles.return_np_array(),
        np.array([[0.0, 0.0, 0.0, 0.2], [0.0, 1.0, 1.0, 0.1], [0.5, 0.5, 0.5, 0.05]]),
    )


def test_atom_initialization_periodic():
    """Tests for periodic atoms"""
    atom_coordinates = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

    atom_radii = {1: 0.2, 2: 0.1, 3: 0.05}
    atom_ids = np.array([1, 3])

    boundary_types = (
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        (pmmoto.BoundaryType.WALL, pmmoto.BoundaryType.WALL),
    )
    sd = pmmoto.initialize((10, 10, 10), boundary_types=boundary_types)

    # Trim within and add periodic but no periodic boundaries!
    particles = pmmoto.particles.initialize_atoms(
        sd,
        atom_coordinates,
        atom_radii,
        atom_ids,
        add_periodic=True,
    )

    np.testing.assert_array_equal(
        particles.return_np_array(),
        np.array([[0.0, 0.0, 0.0, 0.2], [1.0, 0.0, 0.0, 0.2], [0.5, 0.5, 0.5, 0.05]]),
    )


def test_init_sphere_trim():
    """Tests for trimming spheres"""
    sd = pmmoto.initialize((10, 10, 10))
    spheres_in = np.array(
        [
            [0.0, 0.0, 0.0],
            [0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [1.1, 1.1, 1.1],
        ]
    )
    radii_in = np.array([1.0, 0.5, 0.2, 0.5])
    spheres = pmmoto.particles.initialize_spheres(
        sd, spheres_in, radii=radii_in, trim_within=True
    )
    np.testing.assert_array_equal(
        spheres.return_np_array(),
        np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0, 1.0, 1.0, 0.5],
                [0.5, 0.5, 0.5, 0.2],
            ]
        ),
    )
