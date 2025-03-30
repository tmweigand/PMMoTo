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

    bin = pmmoto.analysis.bins.Bin(start=start, end=end, num_bins=num_bins)

    assert bin.width == 0.12

    assert len(bin.centers) == num_bins

    np.testing.assert_array_equal(bin.values, np.zeros(num_bins))

    ones = np.ones(num_bins)
    bin = pmmoto.analysis.bins.Bin(start=start, end=end, num_bins=num_bins, values=ones)

    np.testing.assert_array_equal(bin.values, ones)


def test_bins():
    """
    Test bins
    """

    start = [0, 1]
    end = [3, 2.8]
    num_bins = [25, 50]
    labels = [1, 5]

    bins = pmmoto.analysis.bins.Bins(
        starts=start, ends=end, num_bins=num_bins, labels=labels
    )

    assert bins.bins[1].width == 0.12
    assert bins.bins[5].width == 0.036


def test_count_locations():
    """
    Test binning coordinates
    """

    start = -1
    end = 3
    num_bins = 25

    bin = pmmoto.analysis.bins.Bin(start, end, num_bins)

    coordinates = np.array([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [2.1, 0.0, 0.0]])

    counts = pmmoto.analysis.bins.count_locations(
        coordinates=coordinates, dimension=0, bin=bin
    )

    print(counts)
