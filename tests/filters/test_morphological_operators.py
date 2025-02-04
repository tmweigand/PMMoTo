"""test_morphology.py"""

"""test_edt.py"""

import scipy.ndimage
import numpy as np
import pmmoto
from mpi4py import MPI
import pytest


def test_gen_struct_ratio():
    """
    Convert sphere radius to num of voxels.
    """

    struct_ratio = pmmoto.filters.morphological_operators.gen_struct_ratio(
        resolution=[0.5, 0.02, 1.5], radius=1
    )

    np.testing.assert_equal(struct_ratio, [2, 50, 1])

    struct_ratio = pmmoto.filters.morphological_operators.gen_struct_ratio(
        [0.5, 0.02, 1.5], 10
    )

    np.testing.assert_equal(struct_ratio, [20, 500, 7])


def test_gen_struct_element():
    """
    Generate a spherical/circular structuring element
    """

    _, struct_element = pmmoto.filters.morphological_operators.gen_struct_element(
        resolution=[0.5, 0.02, 0.25], radius=1
    )

    # struct_element shape is equal to 2*struct_ratio + 1
    assert struct_element.shape == (5, 101, 9)
    assert np.sum(struct_element) == 1569

    _, struct_element = pmmoto.filters.morphological_operators.gen_struct_element(
        resolution=[0.05, 0.02, 0.25], radius=0.1
    )

    # struct_element shape is equal to 2*struct_ratio + 1
    assert struct_element.shape == (5, 11, 3)
    assert np.sum(struct_element) == 31


@pytest.mark.mpi(min_size=8)
def test_morphological_addition(generate_simple_subdomain):
    """
    Generate a spherical/circular structuring element
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    radius = 0.1
    subdomains = (2, 2, 2)

    sd = generate_simple_subdomain(
        0,
        periodic=True,
        specified_types=((0, 0), (0, 0), (0, 0)),
        # specified_types=((2, 2), (2, 2), (2, 2)),
        voxels_in=(100, 100, 100),
    )

    img_base = np.ones(sd.domain.voxels).reshape(sd.domain.voxels)
    img_base = pmmoto.domain_generation.gen_smoothed_random_binary_grid(
        sd.domain.voxels, seed=3
    )

    img = pmmoto.core.utils.constant_pad_img(img_base, sd.pad, 4)
    img = pmmoto.core.communication.update_buffer(sd, img)

    add = pmmoto.filters.morphological_operators.addition(
        subdomain=sd,
        img=img,
        radius=radius,
        fft=False,
    )
    add = pmmoto.core.utils.unpad(add, sd.pad)

    _, struct_element = pmmoto.filters.morphological_operators.gen_struct_element(
        resolution=sd.domain.resolution, radius=radius
    )

    # Only valid for wall boundary types
    scipy_add = scipy.ndimage.binary_dilation(img, struct_element)

    np.testing.assert_array_almost_equal(
        add,
        scipy_add,
    )

    sd_local, local_img = pmmoto.core.pmmoto.deconstruct_grid(
        sd,
        img_base,
        subdomains=subdomains,
        rank=rank,
    )

    sd_local, local_add = pmmoto.core.pmmoto.deconstruct_grid(
        sd,
        add,
        subdomains=subdomains,
        rank=rank,
    )

    add_img_edt = pmmoto.filters.morphological_operators.addition(
        subdomain=sd_local, img=local_img, radius=radius, fft=False
    )

    add_img_fft = pmmoto.filters.morphological_operators.addition(
        subdomain=sd_local, img=local_img, radius=radius, fft=True
    )

    np.testing.assert_array_almost_equal(add_img_fft, local_add)
    np.testing.assert_array_almost_equal(add_img_edt, local_add)
    np.testing.assert_array_almost_equal(add_img_edt, add_img_fft)


@pytest.mark.mpi(min_size=8)
def test_morphological_subtraction(generate_simple_subdomain):
    """
    Generate a spherical/circular structuring element
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    subdomains = (2, 2, 2)
    radius = 0.1

    sd = generate_simple_subdomain(
        0,
        periodic=True,
        # specified_types=((2, 2), (2, 2), (2, 2)),
        specified_types=((0, 0), (0, 0), (0, 0)),
        voxels_in=(100, 100, 100),
    )

    # img_base = pmmoto.domain_generation.gen_linear_img(sd.domain.voxels, 2)

    img_base = np.ones(sd.domain.voxels).reshape(sd.domain.voxels)
    img_base = pmmoto.domain_generation.gen_smoothed_random_binary_grid(
        sd.domain.voxels, seed=3
    )

    img = pmmoto.core.utils.constant_pad_img(img_base, sd.pad, 4)
    img = pmmoto.core.communication.update_buffer(sd, img)

    subtract = pmmoto.filters.morphological_operators.subtraction(
        subdomain=sd, img=img, radius=radius, fft=False
    )

    subtract = pmmoto.core.utils.unpad(subtract, sd.pad)

    _, struct_element = pmmoto.filters.morphological_operators.gen_struct_element(
        resolution=sd.domain.resolution, radius=radius
    )

    # Only valid for wall boundary types
    scipy_subtract = scipy.ndimage.binary_erosion(img, struct_element)

    np.testing.assert_array_almost_equal(
        subtract,
        scipy_subtract,
    )

    sd_local, local_img = pmmoto.core.pmmoto.deconstruct_grid(
        sd,
        img_base,
        subdomains=subdomains,
        rank=rank,
    )

    sd_local, local_subtract = pmmoto.core.pmmoto.deconstruct_grid(
        sd,
        subtract,
        subdomains=subdomains,
        rank=rank,
    )

    subtract_img_edt = pmmoto.filters.morphological_operators.subtraction(
        subdomain=sd_local, img=local_img, radius=radius, fft=False
    )

    subtract_img_fft = pmmoto.filters.morphological_operators.subtraction(
        subdomain=sd_local, img=local_img, radius=radius, fft=True
    )

    np.testing.assert_array_almost_equal(subtract_img_fft, local_subtract)
    np.testing.assert_array_almost_equal(subtract_img_edt, local_subtract)
    np.testing.assert_array_almost_equal(subtract_img_edt, subtract_img_fft)
