"""test_average.py"""

import numpy as np
import pmmoto
from mpi4py import MPI
import pytest


def test_linear_1d() -> None:
    """Calculate the average of an image along a given dimension"""
    sd = pmmoto.initialize((10, 10, 10))
    img = pmmoto.domain_generation.gen_img_linear(sd.voxels, 0)

    average_1d = pmmoto.analysis.average.average_image_along_axis(sd, img, 0)

    np.testing.assert_array_equal(average_1d, 4.5 * np.ones((10, 10)))

    average_1d = pmmoto.analysis.average.average_image_along_axis(sd, img, 1)

    # Rows of 0s,1s,...,9s
    np.testing.assert_array_equal(
        average_1d, np.tile(np.arange(10).reshape(-1, 1), (1, 10))
    )

    average_1d = pmmoto.analysis.average.average_image_along_axis(sd, img, 2)
    np.testing.assert_array_equal(
        average_1d, np.tile(np.arange(10).reshape(-1, 1), (1, 10))
    )


def test_linear_2d() -> None:
    """Calculate the average of an image along two dimension"""
    sd = pmmoto.initialize((10, 10, 10))
    img = pmmoto.domain_generation.gen_img_linear(sd.voxels, 0)

    average_2d = pmmoto.analysis.average.average_image_along_axis(sd, img, (0, 1))
    np.testing.assert_array_equal(average_2d, 4.5 * np.ones(10))

    average_2d = pmmoto.analysis.average.average_image_along_axis(sd, img, (1, 2))
    np.testing.assert_array_equal(average_2d, np.arange(10))

    average_2d = pmmoto.analysis.average.average_image_along_axis(sd, img, (0, 2))
    np.testing.assert_array_equal(average_2d, 4.5 * np.ones(10))

    average_2d = pmmoto.analysis.average.average_image_along_axis(sd, img, (2, 0))
    np.testing.assert_array_equal(average_2d, 4.5 * np.ones(10))


@pytest.mark.mpi(min_size=8)
def test_linear_1d_parallel_8() -> None:
    """Calculate the average of an image along a given dimension"""
    comm = MPI.COMM_WORLD

    subdomains = (2, 2, 2)
    sd = pmmoto.initialize((10, 10, 10), subdomains=subdomains, rank=comm.Get_rank())
    global_img = pmmoto.domain_generation.gen_img_linear(sd.domain.voxels, 0)
    sd, img = pmmoto.domain_generation.deconstruct_img(
        sd, global_img, subdomains, sd.rank
    )

    average_1d = pmmoto.analysis.average.average_image_along_axis(sd, img, 0)
    np.testing.assert_array_equal(average_1d, 4.5 * np.ones([10, 10]))

    average_1d = pmmoto.analysis.average.average_image_along_axis(sd, img, 1)
    # Rows of 0s,1s,...,9s
    np.testing.assert_array_equal(
        average_1d, np.tile(np.arange(10).reshape(-1, 1), (1, 10))
    )

    average_1d = pmmoto.analysis.average.average_image_along_axis(sd, img, 2)
    # Rows of 0s,1s,...,9s
    np.testing.assert_array_equal(
        average_1d, np.tile(np.arange(10).reshape(-1, 1), (1, 10))
    )


@pytest.mark.mpi(min_size=8)
def test_linear_2d_parallel_8() -> None:
    """Calculate the average of an image along a given dimension"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    subdomains = (2, 2, 2)
    sd = pmmoto.initialize((10, 10, 10), subdomains=subdomains, rank=rank)
    global_img = pmmoto.domain_generation.gen_img_linear(sd.domain.voxels, 0)
    sd, img = pmmoto.domain_generation.deconstruct_img(
        sd, global_img, subdomains, sd.rank
    )

    average_2d = pmmoto.analysis.average.average_image_along_axis(sd, img, (1, 2))
    np.testing.assert_array_equal(average_2d, np.arange(10))

    average_2d = pmmoto.analysis.average.average_image_along_axis(sd, img, (0, 1))
    np.testing.assert_array_equal(average_2d, 4.5 * np.ones(10))

    average_2d = pmmoto.analysis.average.average_image_along_axis(sd, img, (0, 2))
    np.testing.assert_array_equal(average_2d, 4.5 * np.ones(10))
