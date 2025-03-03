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

    # lattice = pmmoto.domain_generation.lattice_packings.SimpleCubic(sd, 0.025)
    # lattice = pmmoto.domain_generation.lattice_packings.BodyCenteredCubic(sd, 1)
    # lattice = pmmoto.domain_generation.lattice_packings.FaceCenteredCubic(sd, 1)
    # atoms = lattice.generate_lattice()

    atoms = np.load("tests/test_data/atom_data/water.npy")

    # Add radii
    radii = 2 * np.ones((atoms.shape[0], 1))
    atoms = np.hstack((atoms, radii))

    box = [[0, 0], [0, 0], [0, 0]]
    for dim in range(3):
        box[dim][0] = np.min(atoms[:, dim])
        box[dim][1] = np.max(atoms[:, dim])

    print(box, atoms.shape)

    sd = pmmoto.initialize(
        voxels=(10, 10, 10), box=box, boundary_types=((2, 2), (2, 2), (2, 2))
    )

    n_bins = 90
    max_rdf = 18

    probe_atom = np.copy(atoms)
    atoms = pmmoto.domain_generation.gen_periodic_spheres(sd, atoms)

    atoms = pmmoto.domain_generation.rdf.g_rdf(sd, probe_atom, max_rdf, atoms, n_bins)

    # n_bins = 90
    # max_rdf = 5
    # g_r, bin_centers = pmmoto.domain_generation.rdf.local_rdf(
    #     sd, probe_atom, max_rdf, atoms, n_bins
    # )

    # import matplotlib.pyplot as plt

    # plt.plot(bin_centers, g_r)
    # plt.show()

    # pmmoto.io.output.save_img_data_serial(
    #     "data_out/test_particles_subdomain", sd, np.zeros(sd.voxels)
    # )
    pmmoto.io.output.save_particle_data("data_out/test_particles", sd, atoms)

    print(atoms)
