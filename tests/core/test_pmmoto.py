"""test_pmmoto.py"""

import pmmoto


def test_initialization():
    """Test initialization of pmmoto"""
    sd = pmmoto.initialize((10, 10, 10))
    assert isinstance(sd, pmmoto.core.subdomain_verlet.VerletSubdomain)

    sd = pmmoto.initialize((10, 10, 10), return_subdomain=True)
    assert isinstance(sd, pmmoto.core.subdomain.Subdomain)

    sd = pmmoto.initialize((10, 10, 10), verlet_domains=(0, 0, 0))
    assert isinstance(sd, pmmoto.core.subdomain_padded.PaddedSubdomain)
