"""test_porosimetry.py"""

import pytest
import pmmoto
import numpy as np


def test_porosimetry_sizes():
    """
    testing sizes arguement expected behavior
    """
    min_value = 0
    max_value = 10
    num_values = 4
    values = pmmoto.filters.porosimetry.get_sizes(
        min_value, max_value, num_values, "linear"
    )
    # linear test
    np.testing.assert_array_almost_equal(values, [0, 3.33333333, 6.66666667, 10])

    values = pmmoto.filters.porosimetry.get_sizes(
        min_value, max_value, num_values, "log"
    )
    # log test
    np.testing.assert_array_almost_equal_nulp(
        values, [1.000000e00, 2.154435e03, 4.641589e06, 1.000000e10]
    )


@pytest.mark.xfail
def test_porosimetry_sizes_input_fail():
    """
    ensuring checks are behaving correctly
    """

    values = pmmoto.filters.porosimetry.get_sizes(5, 1, 5)
    values = pmmoto.filters.porosimetry.get_sizes(0, 10, 0)
