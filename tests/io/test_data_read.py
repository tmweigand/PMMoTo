"""test_data_read.py"""

import numpy as np
import pmmoto


def test_read_sphere_pack():
    """Test reading of a sphere pack"""
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
    """Test behavior of read atom map"""
    atom_map_file = "tests/test_data/atom_data/atom_map.txt"

    atom_map = pmmoto.io.data_read.read_atom_map(atom_map_file)

    assert atom_map == {
        1: {"element": "H", "name": "BH1"},
    }
