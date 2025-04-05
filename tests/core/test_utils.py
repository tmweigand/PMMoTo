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


def test_bin_image():
    """
    Test for counting occurrences of a value
    """
    N = 10
    sd = pmmoto.initialize((N, N, N))
    img = pmmoto.domain_generation.gen_linear_img(sd.voxels, 0)

    counts = pmmoto.core.utils.bin_image(sd, img)

    assert counts == {
        0.0: 100,
        1.0: 100,
        2.0: 100,
        3.0: 100,
        4.0: 100,
        5.0: 100,
        6.0: 100,
        7.0: 100,
        8.0: 100,
        9.0: 100,
    }


@pytest.mark.mpi(min_size=8)
def test_bin_image_parallel():
    """
    Test for counting occurrences of a value
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    N = 10
    sd = pmmoto.initialize((N, N, N), subdomains=(2, 2, 2), rank=rank)
    img = pmmoto.domain_generation.gen_linear_img(sd.voxels, 0)

    counts = pmmoto.core.utils.bin_image(sd, img, own=True)

    assert counts == {0.0: 100, 1.0: 200, 2.0: 200, 3.0: 200, 4.0: 200, 5.0: 100}

    counts = pmmoto.core.utils.bin_image(sd, img, own=False)

    assert counts == {
        0.0: 288,
        1.0: 288,
        2.0: 288,
        3.0: 288,
        4.0: 288,
        5.0: 288,
    }  # Counting buffer as well - so double counting


def test_check_img_for_solid():
    """
    Ensure solid-0 exists on image
    """
    sd = pmmoto.initialize((10, 10, 10))
    img = np.zeros(sd.voxels)

    pmmoto.core.utils.check_img_for_solid(sd, img)


# @pytest.mark.xfail
def test_check_img_for_solid_fail():
    """
    Ensure solid-0 exists on image
    """
    sd = pmmoto.initialize((10, 10, 10))
    img = np.ones(sd.voxels)

    pmmoto.core.utils.check_img_for_solid(sd, img)
