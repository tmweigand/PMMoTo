"""test_domain.py"""
import numpy as np
import pmmoto

def test_init(domain_data):
    """
    Test for checking initialization of domain values
    """
    domain = pmmoto.core.Domain(
        domain_data['size_domain'],
        domain_data['boundaries'],
        domain_data['inlet'],
        domain_data['outlet']
    )

    np.testing.assert_array_equal(domain.size_domain, domain_data['size_domain'])
    assert(domain.boundaries == domain_data['boundaries'])
    assert(domain.inlet == domain_data['inlet'])
    assert(domain.outlet == domain_data['outlet'])
    assert all(domain.length_domain == domain_data['length_domain'])
    assert (domain.periodic)
