"""test_subdomain_features.py"""

import numpy as np
import pmmoto


def test_subdomain_features(
    domain,
    domain_discretization,
    domain_decomposed,
    subdomains,
    subdomain_features_true,
):
    """
    Test for subdomain features
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

    import pickle

    data_out = {"faces": {}, "edges": {}, "corners": {}}

    for rank in range(pmmoto_decomposed_domain.num_subdomains):
        pmmoto_subdomain = pmmoto.core.Subdomain(
            rank=rank,
            index=subdomains["index"][rank],
            neighbor_ranks=subdomains["neighbor_ranks"][rank],
            box=subdomains["box"][rank],
            boundaries=subdomains["boundaries"][rank],
            inlet=subdomains["inlet"][rank],
            outlet=subdomains["outlet"][rank],
            voxels=subdomains["voxels"][rank],
        )

        data_out["faces"][rank] = pmmoto_subdomain.features["faces"]
        assert (
            subdomain_features_true["faces"][rank] == pmmoto_subdomain.features["faces"]
        )
        data_out["edges"][rank] = pmmoto_subdomain.features["edges"].__dict__
        data_out["corners"][rank] = pmmoto_subdomain.features["corners"].__dict__

    with open(
        "/Users/tim/Desktop/pmmoto/tests/core/test_output/test_subdomain_features.pkl",
        "wb",
    ) as file:  # open a text file
        pickle.dump(data_out, file)  # serialize the list
