"""test_data_read.py"""

import numpy as np
import pmmoto


def test_read_sphere_pack():
    """
    Test reading of a sphere pack
    """
    file_in = "tests/test_data/sphere_packs/bcc.out"
    spheres, domain = pmmoto.io.data_read.read_sphere_pack_xyzr_domain(file_in)

    np.testing.assert_array_equal(
        spheres,
        [
            [0.0, 0.0, 0.0, 0.25],
            [0.0, 0.0, 1.0, 0.25],
            [0.0, 1.0, 0.0, 0.25],
            [1.0, 0.0, 0.0, 0.25],
            [0.0, 1.0, 1.0, 0.25],
            [1.0, 0.0, 1.0, 0.25],
            [1.0, 1.0, 0.0, 0.25],
            [1.0, 1.0, 1.0, 0.25],
        ],
    )

    np.testing.assert_array_equal(domain, ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))


def test_read_atom_map():
    """
    Test behavior of read atom map
    """
    atom_map_file = "tests/test_data/atom_data/atom_map.txt"

    atom_map = pmmoto.io.data_read.read_atom_map(atom_map_file)

    assert atom_map == {1: "BC1", 12: "COOH_C"}


def test_read_rdf():
    """
    Test for checking rdf values
    """
    atom_folder = "tests/test_data/atom_data/"

    atom_map, atom_data = pmmoto.io.data_read.read_rdf(atom_folder)

    assert atom_map == {1: "BC1", 12: "COOH_C"}

    np.testing.assert_equal(atom_data[1].shape, (1000, 3))

    np.testing.assert_equal(atom_data[12].shape, (1000, 3))


def test_read_lammps():
    """
    Test for checking rdf values
    """

    lammps_file = "tests/test_data/LAMMPS/membranedata.100005000"

    membrane_positions, membrane_atom_type, domain_data = (
        pmmoto.io.data_read.py_read_lammps_atoms(lammps_file)
    )

    positions, types, domain, time = pmmoto.io.data_read.read_lammps_atoms(lammps_file)

    np.testing.assert_array_equal(membrane_positions, positions)
    np.testing.assert_array_equal(membrane_atom_type, types)
    np.testing.assert_array_equal(domain_data, domain)

    atom_id_charge_map = {
        (1, 0.6797): 1,
        (1, 0.743425): 2,
        (3, -0.23): 3,
        (3, -0.1956): 4,
        (3, -0.1565): 5,
        (3, 0.014): 6,
        (3, 0.1716): 7,
        (4, -0.587509): 8,
        (5, 0.10745): 9,
        (5, 0.131): 10,
        (5, 0.1816): 11,
        (7, -0.4621): 12,
        (7, -0.398375): 13,
        (8, 0.23105): 14,
        (12, -0.5351): 15,
        (14, 0.4315): 16,
    }

    positions, types, domain, time = pmmoto.io.data_read.read_lammps_atoms(
        lammps_file, atom_id_charge_map
    )
