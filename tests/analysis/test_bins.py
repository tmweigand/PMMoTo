"""test_bins.py"""

import numpy as np
import pmmoto


def test_bin():
    """
    Test bins
    """

    start = 0
    end = 3
    num_bins = 25

    bin = pmmoto.analysis.bins.Bin(start, end, num_bins)

    assert bin.width == 0.12

    assert len(bin.centers) == num_bins

    np.testing.assert_array_equal(bin.values, np.zeros(num_bins))

    ones = np.ones(num_bins)
    bin = pmmoto.analysis.bins.Bin(start, end, num_bins, ones)

    np.testing.assert_array_equal(bin.values, ones)


def test_bins():
    """
    Test bins
    """

    start = [0, 1]
    end = [3, 2.8]
    num_bins = [25, 50]
    labels = [1, 5]

    bins = pmmoto.analysis.bins.Bins(start, end, num_bins, labels)

    assert bins.bins[1].width == 0.12
    assert bins.bins[5].width == 0.036
