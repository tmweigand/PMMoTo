"""test_rdf.py"""

import numpy as np
import pmmoto
import pytest


def test_rdf():
    """
    Test for checking rdf values
    """
    atom_folder = "tests/test_data/atom_data/"
    atom_map, atom_data = pmmoto.io.data_read.read_rdf(atom_folder)

    assert atom_map == {
        1: {"element": "C", "label": "BC1"},
        12: {"element": "C", "label": "COOH_C"},
    }

    rdf = pmmoto.domain_generation.rdf.RDF(
        name="BC1", atom_id=1, r=atom_data[1][:, 0], g=atom_data[1][:, 1]
    )

    interp_g = rdf.g(r=atom_data[rdf.atom_id][:, 0])
    np.testing.assert_allclose(atom_data[rdf.atom_id][:, 1], interp_g)


@pytest.mark.skip
def test_bounded_rdf():
    """
    Test for checking rdf values
    """
    atom_folder = "tests/test_data/atom_data/"
    atom_map, atom_data = pmmoto.io.data_read.read_rdf(atom_folder)
    assert atom_map == {
        1: {"element": "C", "label": "BC1"},
        12: {"element": "C", "label": "COOH_C"},
    }
    rdf = pmmoto.domain_generation.rdf.Bounded_RDF(
        name="BC1", atom_id=1, r=atom_data[1][:, 0], g=atom_data[1][:, 2]
    )

    np.testing.assert_allclose(rdf.bounds, [245, 369])
    assert rdf.r(0.23) == 3.0993256961475946
