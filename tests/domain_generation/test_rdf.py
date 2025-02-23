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

    rdf = pmmoto.domain_generation.rdf.RDF(
        name="BC1", atom_id=1, r=atom_data["BC1"][:, 0], g=atom_data["BC1"][:, 1]
    )

    interp_g = rdf.g(r=atom_data[rdf.name][:, 0])
    np.testing.assert_allclose(atom_data[rdf.name][:, 1], interp_g)


def test_bounded_rdf():
    """
    Test for checking rdf values
    """
    atom_folder = "tests/test_data/atom_data/"
    atom_map, atom_data = pmmoto.io.data_read.read_rdf(atom_folder)
    assert atom_map == {"BC1": 1, "COOH_C": 12}
    rdf = pmmoto.domain_generation.rdf.Bounded_RDF(
        name="BC1", atom_id=1, r=atom_data["BC1"][:, 0], g=atom_data["BC1"][:, 2]
    )

    np.testing.assert_allclose(rdf.bounds, [245, 369])
    assert rdf.r(0.23) == 3.0993256961475946


def test_generate_rdf():
    """
    Test for generating a radial distribution function form atom data
    """
    n_bins = 1000
    n_atoms = 500000
    n_probe_atoms = 1
    atoms_positions = np.random.random([n_atoms + n_probe_atoms, 3])

    n_probe_atoms = 1
    probe_atoms = np.random.random([n_probe_atoms, 3])

    sd = pmmoto.initialize(voxels=(10, 10, 10))

    max_rdf = 1

    rdf = pmmoto.domain_generation.rdf.g_rdf(
        sd, probe_atoms, max_rdf, atoms_positions, n_bins
    )

    import matplotlib.pyplot as plt

    plt.plot(rdf)
    plt.show()

    # print(rdf[0:10])
