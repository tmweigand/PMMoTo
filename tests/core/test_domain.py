"""test_domain.py"""
import numpy as np
import pmmoto

def test_domain(domain):
    """
    Test for checking initialization of domain values
    """
    pmmoto_domain = pmmoto.core.Domain(
        domain['size_domain'],
        domain['boundaries'],
        domain['inlet'],
        domain['outlet']
    )

    np.testing.assert_array_equal(pmmoto_domain.size_domain, domain['size_domain'])
    assert(pmmoto_domain.boundaries == domain['boundaries'])
    assert(pmmoto_domain.inlet == domain['inlet'])
    assert(pmmoto_domain.outlet == domain['outlet'])
    assert all(pmmoto_domain.length_domain == domain['length_domain'])
    assert (pmmoto_domain.periodic)
