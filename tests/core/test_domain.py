"""test_domain.py"""

import numpy as np
import pmmoto


def test_domain():
    """
    Test for checking initialization of domain values
    """

    box = ((77, 100), (-45, 101.21), (-9.0, -3.14159))
    boundary_types = ((0, 0), (1, 1), (2, 2))
    inlet = ((1, 0), (0, 0), (0, 0))
    outlet = ((0, 1), (0, 0), (0, 0))

    pmmoto_domain = pmmoto.core.domain.Domain(
        box=box, boundary_types=boundary_types, inlet=inlet, outlet=outlet
    )

    np.testing.assert_array_equal(pmmoto_domain.box, box)
    assert pmmoto_domain.boundary_types == boundary_types
    assert pmmoto_domain.inlet == inlet
    assert pmmoto_domain.outlet == outlet
    assert pmmoto_domain.length == (23.0, 146.20999999999998, 5.85841)
    assert pmmoto_domain.periodic == True
