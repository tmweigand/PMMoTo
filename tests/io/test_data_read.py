"""test_data_read.py"""

import numpy as np
import pmmoto


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
