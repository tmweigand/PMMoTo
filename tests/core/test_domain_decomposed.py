"""Unit tests for domain decomposition in PMMoTo.

This module tests the correct construction and mapping of decomposed domains.
"""

import numpy as np
import pmmoto


def test_decomposed_domain():
    """Test decomposition of domain"""
    box = ((77, 100), (-45, 101.21), (-9.0, -3.14159))
    boundary_types = ((0, 0), (1, 1), (2, 2))
    inlet = ((1, 0), (0, 0), (0, 0))
    outlet = ((0, 1), (0, 0), (0, 0))
    voxels = (10, 10, 10)
    subdomains = (3, 3, 3)

    pmmoto_domain = pmmoto.core.domain.Domain(
        box=box, boundary_types=boundary_types, inlet=inlet, outlet=outlet
    )

    pmmoto_discretized_domain = (
        pmmoto.core.domain_discretization.DiscretizedDomain.from_domain(
            domain=pmmoto_domain, voxels=voxels
        )
    )

    pmmoto_decomposed_domain = (
        pmmoto.core.domain_decompose.DecomposedDomain.from_discretized_domain(
            discretized_domain=pmmoto_discretized_domain,
            subdomains=subdomains,
        )
    )

    assert pmmoto_decomposed_domain.num_subdomains == 27

    np.testing.assert_array_equal(
        pmmoto_decomposed_domain.map,
        np.array(
            [
                [
                    [-2, -2, -2, -2, -2],
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [-2, -2, -2, -2, -2],
                ],
                [
                    [-2, -2, -2, -2, -2],
                    [2, 0, 1, 2, 0],
                    [5, 3, 4, 5, 3],
                    [8, 6, 7, 8, 6],
                    [-2, -2, -2, -2, -2],
                ],
                [
                    [-2, -2, -2, -2, -2],
                    [11, 9, 10, 11, 9],
                    [14, 12, 13, 14, 12],
                    [17, 15, 16, 17, 15],
                    [-2, -2, -2, -2, -2],
                ],
                [
                    [-2, -2, -2, -2, -2],
                    [20, 18, 19, 20, 18],
                    [23, 21, 22, 23, 21],
                    [26, 24, 25, 26, 24],
                    [-2, -2, -2, -2, -2],
                ],
                [
                    [-2, -2, -2, -2, -2],
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [-2, -2, -2, -2, -2],
                ],
            ]
        ),
    )
