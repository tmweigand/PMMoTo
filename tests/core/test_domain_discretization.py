"""test_domain_discretization.py"""

import numpy as np
import pmmoto


def test_discretized_domain():
    """
    Test for checking initialization of domain values
    """

    box = ((77, 100), (-45, 101.21), (-9.0, -3.14159))
    boundary_types = ((0, 0), (1, 1), (2, 2))
    inlet = ((1, 0), (0, 0), (0, 0))
    outlet = ((0, 1), (0, 0), (0, 0))
    voxels = (10, 10, 10)

    pmmoto_domain = pmmoto.core.Domain(
        box=box, boundary_types=boundary_types, inlet=inlet, outlet=outlet
    )

    pmmoto_discretized_domain = pmmoto.core.DiscretizedDomain.from_domain(
        domain=pmmoto_domain, voxels=voxels
    )

    resolution = pmmoto_discretized_domain.get_resolution()

    np.testing.assert_almost_equal(
        resolution, (2.3, 14.620999999999999, 0.5858410000000001)
    )

    coords = pmmoto_discretized_domain.get_coords(box, voxels, resolution)

    np.testing.assert_array_almost_equal(
        coords[0],
        np.array(
            [78.15, 80.45, 82.75, 85.05, 87.35, 89.65, 91.95, 94.25, 96.55, 98.85]
        ),
    )

    np.testing.assert_array_almost_equal(
        coords[1],
        np.array(
            [
                -37.6895,
                -23.0685,
                -8.4475,
                6.1735,
                20.7945,
                35.4155,
                50.0365,
                64.6575,
                79.2785,
                93.8995,
            ]
        ),
    )

    np.testing.assert_array_almost_equal(
        coords[2],
        np.array(
            [
                -8.70708,
                -8.121239,
                -7.535398,
                -6.949557,
                -6.363716,
                -5.777875,
                -5.192034,
                -4.606193,
                -4.020352,
                -3.434511,
            ]
        ),
    )
