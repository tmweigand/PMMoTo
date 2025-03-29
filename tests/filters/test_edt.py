"""test_edt.py"""

import edt
import numpy as np
from mpi4py import MPI
import pytest
import pmmoto


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
        img=img, start=0, end=img.shape[0], resolution=1.0, num_hull=2, forward=True
    )

    assert hull[0]["height"] == 4.0
    assert hull[0]["vertex"] == 0
    assert hull[0]["range"] == -np.inf

    assert hull[1]["height"] == 0.0
    assert hull[1]["vertex"] == 2
    assert hull[1]["range"] == 0.0

    hull = pmmoto.filters.distance._distance.get_boundary_hull_1d(
        img=img, start=1, end=img.shape[0], resolution=1.0, num_hull=3, forward=True
    )

    assert hull[0]["height"] == 0.0
    assert hull[0]["vertex"] == 2

    hull = pmmoto.filters.distance._distance.get_boundary_hull_1d(
        img=img, start=0, end=img.shape[0], resolution=1.0, num_hull=1, forward=False
    )

    assert hull[0]["height"] == 16.0
    assert hull[0]["vertex"] == 8
    assert hull[0]["range"] == 12.5

    hull = pmmoto.filters.distance._distance.get_boundary_hull_1d(
        img=img, start=0, end=img.shape[0], resolution=1.0, num_hull=4, forward=False
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
        img=img,
        start=0,
        end=img.shape[0] - 1,
        resolution=1.0,
        num_hull=5,
        forward=False,
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
    voxels = (100, 100, 100)
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

    pmmoto.io.output.save_img(
        "data_out/test_periodic_3d",
        img,
        **{
            "edt": edt_pmmoto
            - edt_periodic_img[
                voxels[0] : voxels[0] * 2,
                voxels[1] : voxels[1] * 2,
                voxels[2] : voxels[2] * 2,
            ]
        },
    )

    np.testing.assert_array_equal(
        edt_pmmoto,
        edt_periodic_img[
            voxels[0] : voxels[0] * 2,
            voxels[1] : voxels[1] * 2,
            voxels[2] : voxels[2] * 2,
        ],
    )


@pytest.mark.mpi(min_size=8)
def test_pmmoto_3d_parallel():
    """
    Tests EDT with pmmoto
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    periodic = True
    sd = pmmoto.initialize(voxels=(10, 10, 10), boundary_types=((2, 2), (2, 2), (2, 2)))
    img = np.ones(sd.domain.voxels, dtype=np.uint8)
    img = pmmoto.domain_generation.gen_random_binary_grid(
        sd.domain.voxels,
        p_zero=0.05,
        seed=122,
    )

    subdomains = (2, 2, 2)
    sd_local, local_img = pmmoto.core.pmmoto.deconstruct_grid(
        sd,
        img,
        subdomains=subdomains,
        rank=rank,
    )

    pmmoto_old_edt = pmmoto.filters.distance.edt3d(
        img,
        periodic=[periodic, periodic, periodic],
        resolution=sd.domain.resolution,
    )

    ## Create padded subdomain
    img = pmmoto.core.utils.constant_pad_img(img, sd.pad, -1)
    pmmoto.core.communication.update_buffer(sd, img)
    assert np.min(img) > -1

    _, local_edt_img = pmmoto.core.pmmoto.deconstruct_grid(
        sd, pmmoto_old_edt, subdomains=subdomains, rank=rank
    )

    pmmoto_edt = pmmoto.filters.distance.edt(subdomain=sd_local, img=local_img)

    np.testing.assert_array_almost_equal(
        local_edt_img * local_edt_img, pmmoto_edt * pmmoto_edt
    )


def test_periodic_3d_2():
    """
    Tests for periodic domains
    """

    voxels = (50, 50, 50)  # img = np.ones(voxels, dtype=np.uint8)
    prob_zero = 0.2
    seed = 1246
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

    resolution = (1, 0.32325245, 233)
    # resolution = (1, 1, 1)

    ## Ensure the edt of the img and periodic img are not equal
    edt_periodic_img = edt.edt(periodic_img, anisotropy=resolution)

    edt_pmmoto = pmmoto.filters.distance.edt3d(
        img, periodic=[True, True, True], resolution=resolution
    )

    np.testing.assert_array_almost_equal(
        edt_pmmoto,
        edt_periodic_img[
            voxels[0] : voxels[0] * 2,
            voxels[1] : voxels[1] * 2,
            voxels[2] : voxels[2] * 2,
        ],
    )


def test_edt_single_non_periodic():
    """
    Check to make sure this is called as it needs no correctors. Domain is unit length so max edt must be less than zero.
    Pretty inefficient approach.
    """
    voxels = (10, 10, 10)
    prob_zero = 0.1
    seed = 1
    img = pmmoto.domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)

    sd = pmmoto.initialize(voxels=voxels)
    edt = pmmoto.filters.distance.edt(img, sd)

    assert np.max(edt) < 1.0
