"""test_initialize.py"""

import numpy as np
import pmmoto


def test_initialize(domain, domain_discretization, domain_decomposed):
    """
    Test for checking initialization of domain values
    """
    pmmoto_domain = pmmoto.core.Domain(
        domain["box"], domain["boundaries"], domain["inlet"], domain["outlet"]
    )

    pm = pmmoto.initialize(
        box=domain["box"],
        boundaries=domain["boundaries"],
        inlet=domain["inlet"],
        outlet=domain["outlet"],
        subdomain_map=domain_decomposed["subdomain_map"],
        voxels=domain_discretization["voxels"],
    )
