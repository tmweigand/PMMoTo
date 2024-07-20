"""test_subdomain.py"""

import numpy as np
import pmmoto


def test_subdomain(domain, domain_discretization, subdomain):
    """
    Test for subdomain
    """
    pmmoto_subdomain = pmmoto.core.Subdomain(
        rank=subdomain["rank"],
        index=subdomain["index"],
        box=domain["box"],
        boundaries=domain["boundaries"],
        inlet=domain["inlet"],
        outlet=domain["outlet"],
        num_voxels=domain_discretization["num_voxels"],
    )

    assert pmmoto_subdomain.boundary
