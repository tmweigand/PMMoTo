import pytest
import numpy as np
import pmmoto

def test_discretized_domain(decomposed_domain):
    """
    Test subdomain 
    """
    pmmoto_domain = pmmoto.core.DecomposedDomain(
        decomposed_domain['subdomain_map']
        )
    
    assert(np.min(pmmoto_domain.map) == 0)
    assert(np.max(pmmoto_domain.map) == (pmmoto_domain.num_subdomains - 1) )
    np.testing.assert_array_equal(pmmoto_domain.map.flatten(),np.arange(pmmoto_domain.num_subdomains))
    assert(len(pmmoto_domain.subdomains) == pmmoto_domain.num_subdomains)
