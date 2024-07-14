"""test_subdomain.py"""
import numpy as np
import pmmoto

def test_subdomain(
        domain,
        domain_discretization,
        subdomain
        ):
    """
    Test for subdomain
    """
    pmmoto_subdomain = pmmoto.core.Subdomain(
        rank = subdomain['rank'],
        index = subdomain['index'],
        size_domain = domain['size_domain'],
        boundaries = domain['boundaries'],
        inlet = domain['inlet'],
        outlet = domain['outlet'],
        nodes = domain_discretization['nodes']
    )

    assert(pmmoto_subdomain.boundary)