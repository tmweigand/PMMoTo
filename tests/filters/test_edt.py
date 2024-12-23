"""test_edt.py"""

import numpy as np
import pmmoto
import pytest
import edt


def pretty_print_3d_array(array, slice_dimension=0):
    """
    Nicely prints a 3D array with aligned columns for each 2D slice.

    :param array: 3D list or nested list to print
    :param slice_dimension: The dimension along which to slice the 3D array (0, 1, or 2)
    """

    if slice_dimension == 0:
        slices = array
    elif slice_dimension == 1:
        slices = [list(zip(*plane)) for plane in array]
    elif slice_dimension == 2:
        slices = [list(zip(*slice)) for slice in zip(*array)]
    else:
        raise ValueError("slice_dimension must be 0, 1, or 2")

    for depth_index, slice_2d in enumerate(slices):
        print(f"Slice {depth_index}:")

        # Find the maximum width for each column in the current 2D slice
        col_widths = [max(len(str(item)) for item in col) for col in zip(*slice_2d)]

        # Print each row with formatted columns
        for row in slice_2d:
            print(
                "  ".join(
                    f"{str(item):{col_width}}"
                    for item, col_width in zip(row, col_widths)
                )
            )

        print()  # Add a blank line between slices


def pretty_print_2d_array(array):
    """
    Nicely prints a 2D array with aligned columns.

    :param array: 2D list or nested list to print
    """

    # Find the maximum width for each column
    col_widths = [max(len(str(item)) for item in col) for col in zip(*array)]

    # Print each row with formatted columns
    for row in array:
        print(
            "  ".join(
                f"{str(item):{col_width}}" for item, col_width in zip(row, col_widths)
            )
        )


def test_edt_3d():
    """
    Test the Euclidean transform code for a single process
    """
    img = np.ones([4, 4, 4], dtype=np.uint8)
    img[3, 3, 3] = 0

    out = pmmoto.filters.distance.edt(img)
    true = edt.edt(img)
    np.testing.assert_array_almost_equal(out, true)


def test_edt_2d():
    """
    Test the Euclidean transform code for a single process
    """

    voxels = (10, 50)
    prob_zero = 0.1
    seed = 1
    img = pmmoto.domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)
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


def test_edt_periodic_domains():
    """
    Tests for periodic domains
    """

    ## Generate and test 2d periodic domain in 1-dimension
    voxels = 5
    prob_zero = 0.2
    seed = 1
    img = pmmoto.domain_generation.gen_random_binary_grid(
        (voxels, voxels), prob_zero, seed
    )
    periodic_img = np.tile(img, 3)
    np.testing.assert_array_equal(img, periodic_img[:, voxels : voxels * 2])

    ## Ensure the edt of the img and periodic img are not equal
    edt_img = edt.edt(img)
    edt_periodic_img = edt.edt(periodic_img)
    assert not np.array_equal(edt_img, edt_periodic_img[:, voxels : voxels * 2])

    ## Collect correctors for left and right faces
    left = pmmoto.core.voxels._voxels.get_nearest_boundary_index_face_2d(
        img=img,
        dimension=1,
        label=0,
        forward=True,
    ).astype(np.float32)

    right = pmmoto.core.voxels._voxels.get_nearest_boundary_index_face_2d(
        img=img,
        dimension=1,
        label=0,
        forward=False,
    ).astype(np.float32)

    ## Loop though rows and apply correctors
    edt_corrected = np.zeros_like(img).astype(np.float32)
    count = 0
    for row, l_c, r_c in zip(img, left, right):
        if l_c == -1:
            l_c = np.inf
        if r_c == -1:
            r_c = np.inf
        edt_corrected[count, :] = (
            pmmoto.filters.distance._distance.determine_initial_envelope_1d(
                img=row,
                start=0,
                size=len(row),
                lower_corrector=voxels - r_c,  # swap left and right
                upper_corrector=l_c + 1,  # swap left and right
            )
        )
        count += 1

    pmmoto.filters.distance._distance.get_parabolic_envelope_2d(
        edt_corrected, dimension=0
    )

    edt_corrected = np.sqrt(edt_corrected)
    np.testing.assert_array_almost_equal(
        edt_corrected, edt_periodic_img[:, voxels : voxels * 2]
    )


def test_perioidic_2d():
    """
    Tests for periodic domains
    """

    ## Generate and test 2d periodic domain in 1-dimension
    voxels = (6, 6)
    prob_zero = 0.1
    seed = 2
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

    edt_pmmoto = pmmoto.filters.distance.edt2d(img, periodic=True)

    np.testing.assert_array_almost_equal(
        edt_pmmoto,
        edt_periodic_img[voxels[0] : voxels[0] * 2, voxels[1] : voxels[1] * 2],
    )


def test_perioidic_2d_2():
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

    edt_pmmoto = pmmoto.filters.distance.edt2d(img, periodic=True)

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
    values = pmmoto.filters.distance._distance.get_boundary_hull_1d(
        img=img, start=0, end=img.shape[0], left=True
    )

    assert values == (0, 4.0)

    values = pmmoto.filters.distance._distance.get_boundary_hull_1d(
        img=img, start=1, end=img.shape[0], left=True
    )

    assert values == (1, 0.0)

    values = pmmoto.filters.distance._distance.get_boundary_hull_1d(
        img=img, start=0, end=img.shape[0], left=False
    )

    assert values == (7, 6.0)


def test_boundary_hull_2d():
    """_summary_"""
    voxels = (6, 6)
    prob_zero = 0.7
    seed = 2
    img = pmmoto.domain_generation.gen_random_binary_grid(
        voxels, prob_zero, seed
    ).astype(np.float32)
    periodic_img = np.tile(img, (3, 3))

    values = pmmoto.filters.distance._distance.get_boundary_hull_2d(
        img=periodic_img, dimension=0
    )


def test_perioidic_3d():
    """
    Tests for periodic domains
    """

    ## Generate and test 2d periodic domain in 1-dimension
    voxels = (6, 6, 6)
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

    edt_pmmoto = pmmoto.filters.distance.edt3d(img, periodic=True)

    np.testing.assert_array_equal(
        edt_pmmoto,
        edt_periodic_img[
            voxels[0] : voxels[0] * 2,
            voxels[1] : voxels[1] * 2,
            voxels[2] : voxels[2] * 2,
        ],
    )
