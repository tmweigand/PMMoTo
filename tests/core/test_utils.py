"""test_utils.py"""

import pmmoto
import numpy as np
import pytest
from mpi4py import MPI


def test_decompose_img():
    """Ensure expected behavior of decompose_img"""
    n = 5
    img = np.arange(n * n * n).reshape(n, n, n)
    start = (0, 0, 0)
    shape = (3, 3, 3)
    result = pmmoto.core.utils.decompose_img(img, start=start, shape=shape)

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [[0, 1, 2], [5, 6, 7], [10, 11, 12]],
                [[25, 26, 27], [30, 31, 32], [35, 36, 37]],
                [[50, 51, 52], [55, 56, 57], [60, 61, 62]],
            ]
        ),
    )

    start = (-1, -1, -1)
    shape = (3, 3, 3)
    result = pmmoto.core.utils.decompose_img(img, start=start, shape=shape)

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [[124, 120, 121], [104, 100, 101], [109, 105, 106]],
                [[24, 20, 21], [4, 0, 1], [9, 5, 6]],
                [[49, 45, 46], [29, 25, 26], [34, 30, 31]],
            ]
        ),
    )


def test_decompose_img_2():
    """Ensure expected behavior of decompose_img"""
    n = 10
    linear_values = np.linspace(0, n - 1, n, endpoint=True)
    img = np.ones((n, n, n)) * linear_values
    start = (-1, -1, -1)
    shape = (4, 4, 4)

    result = pmmoto.core.utils.decompose_img(img, start=start, shape=shape)
    expected_result = np.tile(np.array([9.0, 0.0, 1.0, 2.0]), (4, 4, 1))

    np.testing.assert_array_equal(result, expected_result)

    start = (9, 9, 9)
    shape = (4, 4, 4)

    result = pmmoto.core.utils.decompose_img(img, start=start, shape=shape)

    np.testing.assert_array_equal(result, expected_result)


def test_pad():
    """
    Test of padding and un-padding an array
    """
    voxels = (25, 25, 25)

    img = pmmoto.domain_generation.gen_random_binary_grid(voxels)

    pad = ((3, 3), (3, 3), (3, 3))
    pad_img = pmmoto.core.utils.constant_pad_img(img, pad, 8)
    unpad_img = pmmoto.core.utils.unpad(pad_img, pad)
    np.testing.assert_array_equal(img, unpad_img)

    pad = ((0, 3), (0, 0), (3, 0))
    pad_img = pmmoto.core.utils.constant_pad_img(img, pad, 8)
    unpad_img = pmmoto.core.utils.unpad(pad_img, pad)
    np.testing.assert_array_equal(img, unpad_img)


@pytest.mark.mpi(min_size=8)
def test_determine_max():
    """
    Test to ensure we can find the global maximum
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    sd = pmmoto.initialize((10, 10, 10), subdomains=(2, 2, 2), rank=rank)
    img = np.ones(sd.voxels) * sd.rank

    global_max = pmmoto.core.utils.determine_maximum(img)

    assert global_max == 7
