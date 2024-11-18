"""test_subdomain.py"""

import numpy as np
import pmmoto


def test_subdomain():
    """
    Test for subdomain
    """

    box = ((77, 100), (-45, 101.21), (-9.0, -3.14159))
    boundary_types = ((0, 0), (1, 1), (2, 2))
    inlet = ((1, 0), (0, 0), (0, 0))
    outlet = ((0, 1), (0, 0), (0, 0))
    voxels = (10, 10, 10)
    subdomains = (3, 3, 3)

    pmmoto_domain = pmmoto.core.Domain(
        box=box, boundary_types=boundary_types, inlet=inlet, outlet=outlet
    )

    pmmoto_discretized_domain = pmmoto.core.DiscretizedDomain.from_domain(
        domain=pmmoto_domain, voxels=voxels
    )

    pmmoto_decomposed_domain = (
        pmmoto.core.domain_decompose.DecomposedDomain.from_discretized_domain(
            discretized_domain=pmmoto_discretized_domain,
            subdomains=subdomains,
        )
    )

    sd = pmmoto.core.subdomain.Subdomain(
        rank=12, decomposed_domain=pmmoto_decomposed_domain
    )

    print(sd.__dict__)
