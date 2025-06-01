"""test_rdf.py"""

import pmmoto


def test_rdf():
    """Test for checking rdf values
    """
    atom_folder = "tests/test_data/atom_data/"
    atom_map, rdf = pmmoto.io.data_read.read_rdf(atom_folder)

    assert atom_map == {1: {"element": "H", "name": "BH1"}}

    assert rdf[1].interpolate_rdf(3) == 1.1235851734374689


def test_bounded_rdf():
    """Test for checking rdf values
    """
    atom_folder = "tests/test_data/atom_data/"
    atom_map, rdf = pmmoto.io.data_read.read_rdf(atom_folder)
    assert atom_map == {
        1: {"element": "H", "name": "BH1"},
    }

    bounded_rdf = pmmoto.domain_generation.rdf.Bounded_RDF.from_rdf(rdf[1], eps=1.0e-3)

    assert bounded_rdf.interpolate_radius_from_pmf(5.0) == 2.386833861580646
