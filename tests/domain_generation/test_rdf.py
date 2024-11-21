"""test_rdf.py"""

import numpy as np
import pmmoto


def test_rdf():
    """
    Test for checking rdf values
    """
    atom_folder = "tests/test_data/atom_data/"

    atom_map, atom_data = pmmoto.io.data_read.read_rdf(atom_folder)

    assert atom_map == {"BC1": 1, "COOH_C": 12}

    rdf = pmmoto.domain_generation.rdf.RDF("BC1", 1)
    rdf.set_RDF(atom_data[rdf.name][:, 0], atom_data[rdf.name][:, 1])

    interp_g = rdf.g(atom_data[rdf.name][:, 0])

    np.testing.assert_allclose(atom_data[rdf.name][:, 1], interp_g)


def test_bounded_rdf():
    """
    Test for checking rdf values
    """
    atom_folder = "tests/test_data/atom_data/"

    atom_map, atom_data = pmmoto.io.data_read.read_rdf(atom_folder)

    assert atom_map == {"BC1": 1, "COOH_C": 12}

    rdf = pmmoto.domain_generation.rdf.RDF("BC1", 1)
    rdf.set_RDF(atom_data[rdf.name][:, 0], atom_data[rdf.name][:, 1])

    interp_g = rdf.g(atom_data[rdf.name][:, 0])

    np.testing.assert_allclose(atom_data[rdf.name][:, 1], interp_g)
