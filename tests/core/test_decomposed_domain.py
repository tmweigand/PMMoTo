import pytest
import numpy as np
import pmmoto

@pytest.fixture
def decomposed_domain_data():
    """
    Subdomain data to pass into tests
    """
    data = {
        'subdomains': (2,2,2)
    }

    return data

def test_discretized_domain(decomposed_domain_data):
    """
    Test subdomain 
    """
    pmmoto_domain = pmmoto.core.DecomposedDomain(decomposed_domain_data['subdomains'])
    
    assert(np.min(pmmoto_domain.map) == 0)
    assert(np.max(pmmoto_domain.map) == (pmmoto_domain.num_subdomains - 1) )
    np.testing.assert_array_equal(pmmoto_domain.map.flatten(),np.arange(pmmoto_domain.num_subdomains))
    assert(len(pmmoto_domain.subdomains) == pmmoto_domain.num_subdomains)
