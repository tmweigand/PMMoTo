"""test_domain_discretization.py"""

import pmmoto


def test_discretized_domain(domain, domain_discretization, domain_discretization_true):
    """
    Test for checking initialization of domain values
    """
    pmmoto_domain = pmmoto.core.Domain(
        domain["box"], domain["boundaries"], domain["inlet"], domain["outlet"]
    )

    pmmoto_discretized_domain = pmmoto.core.DiscretizedDomain.from_domain(
        domain=pmmoto_domain,
        voxels=domain_discretization["voxels"],
    )

    coords = pmmoto_discretized_domain.get_coords()

    assert pmmoto_discretized_domain.voxels == domain_discretization_true["num_voxels"]
    assert (
        pmmoto_discretized_domain.resolution == domain_discretization_true["resolution"]
    )
    assert all(coords[0] == domain_discretization_true["coords"][0])
    assert all(coords[1] == domain_discretization_true["coords"][1])
    assert all(coords[2] == domain_discretization_true["coords"][2])

    # import pickle

    # data_out = {
    #     "num_voxels": pmmoto_discretized_domain.num_voxels,
    #     "resolution": pmmoto_discretized_domain.resolution,
    #     "coords": pmmoto_discretized_domain.get_coords(),
    # }
    # with open(
    #     "/Users/tim/Desktop/pmmoto/tests/core/test_output/test_discretized_domain.pkl",
    #     "wb",
    # ) as file:  # open a text file
    #     pickle.dump(data_out, file)  # serialize the list
