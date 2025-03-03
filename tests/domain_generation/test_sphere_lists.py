"""test_sphere_lists.py"""

import pytest
import numpy as np
import pmmoto


def test_trim_sphere_list():
    """
    Test selection and re-creation of sphere list
    """
    sd = pmmoto.initialize(voxels=(10, 10, 10))

    # Delete first sphere
    spheres = np.array([[10.5, 0.5, 0.5, 0.25], [0.5, 0.5, 0.5, 0.25]])

    sphere_list = pmmoto.domain_generation._domain_generation.trim_list(sd, spheres)

    np.testing.assert_array_equal(sphere_list.flatten(), spheres[1])

    # No deletions
    spheres = np.array([[1.5, 0.5, 0.5, 0.25], [0.5, 0.5, 0.5, 0.25]])
    sphere_list = pmmoto.domain_generation._domain_generation.trim_list(sd, spheres)

    np.testing.assert_array_equal(sphere_list, spheres)
