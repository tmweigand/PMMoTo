"""test_subdomain_features.py"""

import numpy as np
import pmmoto
import unittest


def compare_dicts(true_dict, sample_dict, rank, feature):
    """Loop through dictionary to compare if equal

    Args:
        true_dict (dict): The true solution
        sample_dict (dict): The test solution
        rank (int): rank to test
        feature (str): "faces" "edges" or "corner"
    """
    for feat in true_dict[feature][rank]:
        for key in true_dict[feature][rank][feat].__dict__:
            true_value = getattr(true_dict[feature][rank][feat], key)
            _type = type(getattr(true_dict[feature][rank][feat], key))
            if _type != dict:
                assert true_value == getattr(sample_dict[feature][feat], key)
            else:
                sample_value = getattr(sample_dict[feature][feat], key)

                for key_2 in true_value.keys():
                    type_2 = type(true_value[key_2])
                    if type_2 != np.ndarray:
                        assert true_value[key_2] == sample_value[key_2]
                    else:
                        assert all(true_value[key_2] == sample_value[key_2])


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

    # import pickle

    # data_out = {"faces": {}, "edges": {}, "corners": {}}

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
            start=subdomains["start"][rank],
            num_subdomains=pmmoto_decomposed_domain.num_subdomains,
            domain_voxels=domain_discretization["voxels"],
        )
        compare_dicts(subdomain_features_true, pmmoto_subdomain.features, rank, "faces")
        compare_dicts(subdomain_features_true, pmmoto_subdomain.features, rank, "edges")
        compare_dicts(
            subdomain_features_true, pmmoto_subdomain.features, rank, "corners"
        )

    #     data_out["faces"][rank] = pmmoto_subdomain.features["faces"]
    #     data_out["edges"][rank] = pmmoto_subdomain.features["edges"]
    #     data_out["corners"][rank] = pmmoto_subdomain.features["corners"]

    # with open(
    #     "/Users/tim/Desktop/pmmoto/tests/core/test_output/test_subdomain_features.pkl",
    #     "wb",
    # ) as file:  # open a text file
    #     pickle.dump(data_out, file)  # serialize the list
