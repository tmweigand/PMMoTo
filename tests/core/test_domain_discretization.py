"""test_domain_discretization.py"""

import pytest
import numpy as np
import pmmoto

@pytest.fixture
def domain_discretization_data():
    """
    Domain discretization data to pass into tests.
    """
    data = {
        'subdomain_map':(2,2,2),
        'nodes':(100,100,100)
    }

    return data


def test_init(domain_discretization_data,domain_data):
    """
    Test for checking initialization of domain values
    """
    domain = pmmoto.core.Domain(
        domain_data['size_domain'],
        domain_data['boundaries'],
        domain_data['inlet'],
        domain_data['outlet']
    )

    discretized_domain = pmmoto.core.DiscretizedDomain(
        domain,
        domain_discretization_data['subdomain_map'],
        domain_discretization_data['nodes']
    )

    assert all(discretized_domain.voxel == 1)
    assert all(discretized_domain.sd_nodes == 50)
    assert all(discretized_domain.rem_nodes == 0)