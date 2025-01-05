"""test_edt.py"""

import scipy.ndimage
import edt
import numpy as np
import pmmoto
from mpi4py import MPI
import pytest


def test_edt_2d():
    """
    Test the Euclidean transform code for a single process and non-periodic domain
    """

    voxels = (10, 50)
    prob_zero = 0.1
    seed = 1
    img = pmmoto.domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)
    out = pmmoto.filters.distance.edt(img)
    true = edt.edt(img)
    np.testing.assert_array_almost_equal(out, true)


def test_edt_3d():
    """
    Test the Euclidean transform code for a single process and non-periodic domain
    """
    img = np.ones([4, 4, 4], dtype=np.uint8)
    img[3, 3, 3] = 0

    out = pmmoto.filters.distance.edt(img)
    true = edt.edt(img)
    np.testing.assert_array_almost_equal(out, true)


def test_initial_parabolic_envelope():
    """
    Test the initial distance sweep with full array
    """
    img = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1], dtype=np.uint8)

    output = pmmoto.filters.distance._distance.determine_initial_envelope_1d(
        img=img, start=0, size=len(img), lower_corrector=np.inf, upper_corrector=np.inf
    )

    np.testing.assert_array_equal(
        output,
        [4.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 4.0, 9.0, 9.0, 4.0, 1.0, 0.0, 1.0],
    )


def test_initial_parabolic_envelope_correctors():
    """
    Test the initial distance sweep with correctors
    """
    img = np.array([1, 1, 0, 1, 1, 1, 1, 1, 1], dtype=np.uint8)

    output = pmmoto.filters.distance._distance.determine_initial_envelope_1d(
        img=img, start=0, size=len(img), lower_corrector=1, upper_corrector=1
    )
    np.testing.assert_array_equal(output, [1.0, 1.0, 0.0, 1.0, 4.0, 9.0, 9.0, 4.0, 1.0])

    img = np.array([1, 1, 0, 1, 1, 1, 1, 1, 1], dtype=np.uint8)

    output = pmmoto.filters.distance._distance.determine_initial_envelope_1d(
        img=img, start=0, size=len(img), lower_corrector=2, upper_corrector=4
    )
    np.testing.assert_array_equal(
        output, [4.0, 1.0, 0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 16.0]
    )


def test_periodic_2d():
    """
    Tests for periodic domains
    """

    import matplotlib.pyplot as plt

    ## Generate and test 2d periodic domain in 1-dimension
    voxels = (60, 60)
    prob_zero = 0.1
    seed = 4
    img = pmmoto.domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)
    img = pmmoto.domain_generation.gen_smoothed_random_binary_grid(
        voxels, prob_zero, smoothness=1, seed=seed
    ).astype(np.uint8)

    periodic_img = np.tile(img, (3, 3)).astype(np.uint8)

    np.testing.assert_array_equal(
        img, periodic_img[voxels[0] : voxels[0] * 2, voxels[1] : voxels[1] * 2]
    )

    edt_periodic_img = edt.edt(periodic_img)

    edt_pmmoto = pmmoto.filters.distance.edt2d(img, periodic=[True, True])

    np.testing.assert_array_almost_equal(
        edt_pmmoto,
        edt_periodic_img[voxels[0] : voxels[0] * 2, voxels[1] : voxels[1] * 2],
    )


def test_periodic_2d_2():
    """
    Tests for periodic domains
    """

    ## Generate and test 2d periodic domain in 1-dimension
    voxels = (6, 6)
    prob_zero = 0.1
    seed = 5
    img = pmmoto.domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)
    periodic_img = np.tile(img, (3, 3))

    np.testing.assert_array_equal(
        img, periodic_img[voxels[0] : voxels[0] * 2, voxels[1] : voxels[1] * 2]
    )

    ## Ensure the edt of the img and periodic img are not equal
    edt_img = edt.edt(img)
    edt_periodic_img = edt.edt(periodic_img)
    assert not np.array_equal(
        edt_img, edt_periodic_img[voxels[0] : voxels[0] * 2, voxels[1] : voxels[1] * 2]
    )

    edt_pmmoto = pmmoto.filters.distance.edt2d(img, periodic=[True, True])

    np.testing.assert_array_almost_equal(
        edt_pmmoto,
        edt_periodic_img[voxels[0] : voxels[0] * 2, voxels[1] : voxels[1] * 2],
    )


def test_periodic_2d_3():
    """
    Tests for periodic domains
    """
    import matplotlib.pyplot as plt

    ## Generate and test 2d periodic domain in 1-dimension
    voxels = (500, 500)
    prob_zero = 0.2
    seed = 286565
    img = pmmoto.domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)
    periodic_img = np.tile(img, (3, 3))

    np.testing.assert_array_equal(
        img, periodic_img[voxels[0] : voxels[0] * 2, voxels[1] : voxels[1] * 2]
    )

    ## Ensure the edt of the img and periodic img are not equal
    edt_img = edt.edt(img)
    edt_periodic_img = edt.edt(periodic_img)
    assert not np.array_equal(
        edt_img, edt_periodic_img[voxels[0] : voxels[0] * 2, voxels[1] : voxels[1] * 2]
    )

    edt_pmmoto = pmmoto.filters.distance.edt2d(img, periodic=[True, True])

    np.testing.assert_array_almost_equal(
        edt_pmmoto,
        edt_periodic_img[voxels[0] : voxels[0] * 2, voxels[1] : voxels[1] * 2],
    )


def test_boundary_hull_1d():
    """_summary_"""
    img = np.array(
        [4, np.inf, 0, np.inf, np.inf, np.inf, 4, 6, 16],
        dtype=np.float32,
    )
    hull = pmmoto.filters.distance._distance.get_boundary_hull_1d(
        img=img, start=0, end=img.shape[0], num_hull=2, left=True
    )

    assert hull[0]["height"] == 4.0
    assert hull[0]["vertex"] == 0
    assert hull[0]["range"] == -np.inf

    assert hull[1]["height"] == 0.0
    assert hull[1]["vertex"] == 2
    assert hull[1]["range"] == 0.0

    hull = pmmoto.filters.distance._distance.get_boundary_hull_1d(
        img=img, start=1, end=img.shape[0], num_hull=3, left=True
    )

    assert hull[0]["height"] == 0.0
    assert hull[0]["vertex"] == 1

    hull = pmmoto.filters.distance._distance.get_boundary_hull_1d(
        img=img, start=0, end=img.shape[0], num_hull=1, left=False
    )

    assert hull[0]["height"] == 16.0
    assert hull[0]["vertex"] == 8
    assert hull[0]["range"] == 12.5

    hull = pmmoto.filters.distance._distance.get_boundary_hull_1d(
        img=img, start=0, end=img.shape[0], num_hull=4, left=False
    )

    assert hull[0]["height"] == 16.0
    assert hull[0]["vertex"] == 8
    assert hull[0]["range"] == 12.5

    assert hull[1]["height"] == 6.0
    assert hull[1]["vertex"] == 7
    assert hull[1]["range"] == 7.5

    assert hull[2]["height"] == 4.0
    assert hull[2]["vertex"] == 6
    assert hull[2]["range"] == 4.5

    assert hull[3]["height"] == 0.0
    assert hull[3]["vertex"] == 2
    assert hull[3]["range"] == 0.0

    hull = pmmoto.filters.distance._distance.get_boundary_hull_1d(
        img=img, start=0, end=img.shape[0] - 1, num_hull=5, left=False
    )

    assert hull[0]["height"] == 6.0
    assert hull[0]["vertex"] == 7
    assert hull[0]["range"] == 7.5

    assert hull[1]["height"] == 4.0
    assert hull[1]["vertex"] == 6
    assert hull[1]["range"] == 4.5

    assert hull[2]["height"] == 0.0
    assert hull[2]["vertex"] == 2
    assert hull[2]["range"] == 0.0


def test_periodic_3d():
    """
    Tests for periodic domains
    """

    ## Generate and test 2d periodic domain in 1-dimension
    voxels = (50, 50, 50)
    prob_zero = 0.1
    seed = 1
    img = pmmoto.domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)
    periodic_img = np.tile(img, (3, 3, 3))

    np.testing.assert_array_equal(
        img,
        periodic_img[
            voxels[0] : voxels[0] * 2,
            voxels[1] : voxels[1] * 2,
            voxels[2] : voxels[2] * 2,
        ],
    )

    ## Ensure the edt of the img and periodic img are not equal
    edt_img = edt.edt(img)
    edt_periodic_img = edt.edt(periodic_img)
    assert not np.array_equal(
        edt_img,
        edt_periodic_img[
            voxels[0] : voxels[0] * 2,
            voxels[1] : voxels[1] * 2,
            voxels[2] : voxels[2] * 2,
        ],
    )

    edt_pmmoto = pmmoto.filters.distance.edt3d(img, periodic=[True, True, True])

    np.testing.assert_array_equal(
        edt_pmmoto,
        edt_periodic_img[
            voxels[0] : voxels[0] * 2,
            voxels[1] : voxels[1] * 2,
            voxels[2] : voxels[2] * 2,
        ],
    )


def test_pmmoto_3d(generate_simple_subdomain):
    """
    Tests EDT with pmmoto
    """
    sd = generate_simple_subdomain(0)
    prob_zero = 0.2
    seed = 124
    img = pmmoto.domain_generation.gen_random_binary_grid(sd.voxels, prob_zero, seed)
    periodic_img = np.tile(img, (3, 3, 3))

    np.testing.assert_array_equal(
        img,
        periodic_img[
            sd.voxels[0] : sd.voxels[0] * 2,
            sd.voxels[1] : sd.voxels[1] * 2,
            sd.voxels[2] : sd.voxels[2] * 2,
        ],
    )

    pmmoto_edt = pmmoto.filters.distance.edt(subdomain=sd, img=img)
    pmmoto_old_edt = pmmoto.filters.distance.edt3d(img, periodic=[True, True, True])
    other_edt = edt.edt(periodic_img)
    np.testing.assert_array_equal(pmmoto_old_edt, pmmoto_edt)

    np.testing.assert_array_equal(
        pmmoto_old_edt,
        other_edt[
            sd.voxels[0] : sd.voxels[0] * 2,
            sd.voxels[1] : sd.voxels[1] * 2,
            sd.voxels[2] : sd.voxels[2] * 2,
        ],
    )


def test_periodic_3d():
    """
    Tests for periodic domains
    """

    ## Generate and test 2d periodic domain in 1-dimension
    voxels = (50, 50, 50)
    prob_zero = 0.1
    seed = 1
    img = pmmoto.domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)
    periodic_img = np.tile(img, (3, 3, 3))

    np.testing.assert_array_equal(
        img,
        periodic_img[
            voxels[0] : voxels[0] * 2,
            voxels[1] : voxels[1] * 2,
            voxels[2] : voxels[2] * 2,
        ],
    )

    ## Ensure the edt of the img and periodic img are not equal
    edt_img = edt.edt(img)
    edt_periodic_img = edt.edt(periodic_img)
    assert not np.array_equal(
        edt_img,
        edt_periodic_img[
            voxels[0] : voxels[0] * 2,
            voxels[1] : voxels[1] * 2,
            voxels[2] : voxels[2] * 2,
        ],
    )

    edt_pmmoto = pmmoto.filters.distance.edt3d(img, periodic=[True, True, True])

    np.testing.assert_array_equal(
        edt_pmmoto,
        edt_periodic_img[
            voxels[0] : voxels[0] * 2,
            voxels[1] : voxels[1] * 2,
            voxels[2] : voxels[2] * 2,
        ],
    )


# @pytest.mark.mpi
def test_pmmoto_3d_parallel(generate_subdomain, generate_simple_subdomain):
    """
    Tests EDT with pmmoto
    """
    sd = generate_simple_subdomain(0)
    prob_zero = 0.2
    seed = 124
    img = pmmoto.domain_generation.gen_random_binary_grid(sd.voxels, prob_zero, seed)
    pmmoto_edt = pmmoto.filters.distance.edt(subdomain=sd, img=img)
    pmmoto_old_edt = pmmoto.filters.distance.edt3d(img, periodic=[True, True, True])
    np.testing.assert_array_equal(pmmoto_old_edt, pmmoto_edt)

    sd = generate_subdomain(0)
    test = pmmoto.core.utils.deconstruct_grid(img)
    print(test)
