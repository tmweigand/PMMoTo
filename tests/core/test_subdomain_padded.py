"""test_subdomain_padded.py"""

import numpy as np
import pmmoto


def test_subdomain_padded(
    domain,
    domain_discretization,
    domain_decomposed,
    subdomains,
    subdomains_padded,
    subdomains_padded_true,
):
    """
    Test for subdomain
    """

    pmmoto_domain = pmmoto.core.Domain(
        domain["box"], domain["boundaries"], domain["inlet"], domain["outlet"]
    )

    pmmoto_discretized_domain = pmmoto.core.DiscretizedDomain.from_domain(
        domain=pmmoto_domain,
        voxels=domain_discretization["voxels"],
    )

    pmmoto_decomposed_domain = (
        pmmoto.core.domain_decompose.DecomposedDomain.from_discretized_domain(
            discretized_domain=pmmoto_discretized_domain,
            subdomain_map=domain_decomposed["subdomain_map"],
        )
    )

    # import pickle

    # data_out = {"pad": {}, "reservoir_pad": {}, "voxels": {}, "box": {}, "coords": {}, "start": {}, "num_subdomain":{}, "domain_voxels":{}}

    for rank in range(pmmoto_decomposed_domain.num_subdomains):
        pmmoto_subdomain = pmmoto.core.Subdomain(
            rank=rank,
            index=subdomains["index"][rank],
            box=subdomains["box"][rank],
            boundaries=subdomains["boundaries"][rank],
            inlet=subdomains["inlet"][rank],
            outlet=subdomains["outlet"][rank],
            voxels=subdomains["voxels"][rank],
            start=subdomains["start"][rank],
            num_subdomains=pmmoto_decomposed_domain.num_subdomains,
            domain_voxels=domain_discretization["voxels"],
            neighbor_ranks=subdomains["neighbor_ranks"][rank],
        )

        pmmoto_padded_subdomain = pmmoto.core.PaddedSubdomain.from_subdomain(
            subdomain=pmmoto_subdomain,
            pad=subdomains_padded["pad"],
            reservoir_voxels=subdomains_padded["reservoir_voxels"],
        )

        np.testing.assert_array_equal(
            pmmoto_padded_subdomain.pad, subdomains_padded_true["pad"][rank]
        )

        np.testing.assert_array_equal(
            pmmoto_padded_subdomain.voxels, subdomains_padded_true["voxels"][rank]
        )

        np.testing.assert_array_equal(
            pmmoto_padded_subdomain.box, subdomains_padded_true["box"][rank]
        )

        np.testing.assert_array_equal(
            pmmoto_padded_subdomain.coords[0], subdomains_padded_true["coords"][rank][0]
        )

        np.testing.assert_array_equal(
            pmmoto_padded_subdomain.coords[1], subdomains_padded_true["coords"][rank][1]
        )

        np.testing.assert_array_equal(
            pmmoto_padded_subdomain.coords[2], subdomains_padded_true["coords"][rank][2]
        )

        # data_out["pad"][rank] = pmmoto_padded_subdomain.pad
        # data_out["reservoir_pad"][rank] = pmmoto_padded_subdomain.reservoir_pad
        # data_out["voxels"][rank] = pmmoto_padded_subdomain.voxels
        # data_out["box"][rank] = pmmoto_padded_subdomain.box
        # data_out["coords"][rank] = pmmoto_padded_subdomain.coords

    # with open(
    #     "/Users/tim/Desktop/pmmoto/tests/core/test_output/test_subdomain_padded.pkl",
    #     "wb",
    # ) as file:  # open a text file
    #     pickle.dump(data_out, file)  # serialize the list
