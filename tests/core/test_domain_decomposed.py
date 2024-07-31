import pytest
import numpy as np
import pmmoto


def test_decomposed_domain(domain, domain_discretization, domain_decomposed):
    """
    Test decomposition of domain
    """

    pmmoto_decomposed_domain = pmmoto.core.DecomposedDomain(
        num_voxels=domain_discretization["num_voxels"],
        box=domain["box"],
        boundaries=domain["boundaries"],
        inlet=domain["inlet"],
        outlet=domain["outlet"],
        subdomain_map=domain_decomposed["subdomain_map"],
    )

    pmmoto_index = pmmoto_decomposed_domain.get_subdomain_index(0)
    voxels = pmmoto_decomposed_domain.get_subdomain_voxels(pmmoto_index)
    boundaries = pmmoto_decomposed_domain.get_subdomain_boundaries(pmmoto_index)
    box = pmmoto_decomposed_domain.get_subdomain_box(pmmoto_index, voxels)
    inlet = pmmoto_decomposed_domain.get_subdomain_inlet(pmmoto_index)
    outlet = pmmoto_decomposed_domain.get_subdomain_outlet(pmmoto_index)

    np.testing.assert_array_equal(
        pmmoto_decomposed_domain.map.flatten(),
        np.arange(pmmoto_decomposed_domain.num_subdomains),
    )

    assert pmmoto_index == (0, 0, 0)
    np.testing.assert_array_equal(boundaries, np.array([[0, -1], [1, -1], [2, -1]]))
    np.testing.assert_array_equal(
        box, np.array([[0.0, 50.0], [0.0, 50.0], [0.0, 50.0]])
    )
    np.testing.assert_array_equal(inlet, np.array([[1, 0], [0, 0], [0, 0]]))
    np.testing.assert_array_equal(outlet, np.array([[0, 0], [0, 0], [0, 0]]))
