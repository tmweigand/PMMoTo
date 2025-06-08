"""test_voxels.py"""

import numpy as np
import pmmoto
from pmmoto.core import _voxels
import pytest


def test_voxls_get_id() -> None:
    """Simple test to check voxel id mapping."""
    x = (1, 2, 7)
    v = (5, 6, 5)
    id = pmmoto.core.voxels.get_id(x, v)
    assert id == 42

    x = (-1, -1, -1)
    v = (10, 10, 10)
    id = pmmoto.core.voxels.get_id(x, v)
    assert id == 999


def test_1d_slice_extraction() -> None:
    """Test for ensuring looping through 1d slice is working with c++ interface.

    For the 2 direction, check on the diagonal
    """
    # Example data
    n = 5
    img = np.arange(n**3, dtype=np.uint8).reshape(n, n, n)

    # Check slice in 0 direction
    out = []
    for y in range(n):
        out.append(
            _voxels.extract_1d_slice(
                img=img,
                dimension=0,
                start=np.array([0, y, 0], dtype=np.uint64),
                forward=True,
            )
        )

    np.testing.assert_equal(
        out,
        [
            [0, 25, 50, 75, 100],
            [5, 30, 55, 80, 105],
            [10, 35, 60, 85, 110],
            [15, 40, 65, 90, 115],
            [20, 45, 70, 95, 120],
        ],
    )

    # Check slice in 0 direction going backwards
    out = []
    for y in range(n):
        out.append(
            _voxels.extract_1d_slice(
                img=img,
                dimension=0,
                start=np.array([0, y, 0], dtype=np.uint64),
                forward=False,
            )
        )
    np.testing.assert_equal(
        out,
        [
            [100, 75, 50, 25, 0],
            [105, 80, 55, 30, 5],
            [110, 85, 60, 35, 10],
            [115, 90, 65, 40, 15],
            [120, 95, 70, 45, 20],
        ],
    )

    # Check slice in 1 direction
    out = []
    for x in range(n):
        out.append(
            _voxels.extract_1d_slice(
                img=img,
                dimension=1,
                start=np.array([x, 0, 0], dtype=np.uint64),
                forward=True,
            )
        )

    np.testing.assert_equal(
        out,
        [
            [0, 5, 10, 15, 20],
            [25, 30, 35, 40, 45],
            [50, 55, 60, 65, 70],
            [75, 80, 85, 90, 95],
            [100, 105, 110, 115, 120],
        ],
    )

    # Check slice in 1 direction going backwards
    out = []
    for x in range(n):
        out.append(
            _voxels.extract_1d_slice(
                img=img,
                dimension=1,
                start=np.array([x, 0, 0], dtype=np.uint64),
                forward=False,
            )
        )
    np.testing.assert_equal(
        out,
        [
            [20, 15, 10, 5, 0],
            [45, 40, 35, 30, 25],
            [70, 65, 60, 55, 50],
            [95, 90, 85, 80, 75],
            [120, 115, 110, 105, 100],
        ],
    )

    # Check slice in 2 direction
    out = []
    for x in range(n):
        out.append(
            _voxels.extract_1d_slice(
                img=img,
                dimension=2,
                start=np.array([x, x, 0], dtype=np.uint64),
                forward=True,
            )
        )

    np.testing.assert_equal(
        out,
        [
            [0, 1, 2, 3, 4],
            [30, 31, 32, 33, 34],
            [60, 61, 62, 63, 64],
            [90, 91, 92, 93, 94],
            [120, 121, 122, 123, 124],
        ],
    )

    # Check slice in 2 direction going backwards
    out = []
    for x in range(n):
        out.append(
            _voxels.extract_1d_slice(
                img=img,
                dimension=2,
                start=np.array([x, x, 0], dtype=np.uint64),
                forward=False,
            )
        )
    np.testing.assert_equal(
        out,
        [
            [4, 3, 2, 1, 0],
            [34, 33, 32, 31, 30],
            [64, 63, 62, 61, 60],
            [94, 93, 92, 91, 90],
            [124, 123, 122, 121, 120],
        ],
    )


def test_get_nearest_boundary_index_1d() -> None:
    """Test for ensuring get_nearest_boundary_index works in 1d.

    Same c++ function is called in 3d so simpler for testing.
    """
    # Example data
    img = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1], dtype=np.uint8)

    index = _voxels.determine_index_nearest_boundary_1d(img=img, label=0, forward=True)

    assert index == 9

    index = _voxels.determine_index_nearest_boundary_1d(img=img, label=0, forward=False)

    assert index == 12


def test_get_nearest_boundary_index_1d_pad() -> None:
    """Adding ability to "de"-pad the image for the 1d case"""
    # Example data
    img = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1], dtype=np.uint8)

    index = _voxels.determine_index_nearest_boundary_1d(
        img=img, label=0, forward=True, start=1, upper_skip=0
    )

    assert index == 9

    img = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
    index = _voxels.determine_index_nearest_boundary_1d(
        img=img, label=0, forward=False, start=0, upper_skip=0
    )
    assert index == 12

    index = pmmoto.core._voxels.determine_index_nearest_boundary_1d(
        img=img, label=0, forward=False, start=1, upper_skip=0
    )
    assert index == 12

    img = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1, 0], dtype=np.uint8)

    index = pmmoto.core._voxels.determine_index_nearest_boundary_1d(
        img=img, label=0, forward=False, start=0, upper_skip=0
    )
    assert index == 9

    index = _voxels.determine_index_nearest_boundary_1d(
        img=img, label=0, forward=False, start=1, upper_skip=0
    )
    assert index == 9

    img = np.array([0, 1, 0, 0, 0, 0, 1, 1, 1, 0], dtype=np.uint8)

    index = _voxels.determine_index_nearest_boundary_1d(
        img=img, label=0, forward=True, start=0, upper_skip=0
    )
    assert index == 0

    index = _voxels.determine_index_nearest_boundary_1d(
        img=img, label=0, forward=True, start=1, upper_skip=0
    )
    assert index == 2

    index = _voxels.determine_index_nearest_boundary_1d(
        img=img, label=0, forward=False, start=0, upper_skip=0
    )
    assert index == 9

    index = _voxels.determine_index_nearest_boundary_1d(
        img=img, label=0, forward=False, start=0, upper_skip=1
    )
    assert index == 5

    img = np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
    index = _voxels.determine_index_nearest_boundary_1d(
        img=img, label=0, forward=False, start=1, upper_skip=1
    )
    assert index == 1


def test_get_nearest_boundary_index() -> None:
    """Test for ensuring get_nearest_boundary_index works. duh"""
    # Example data
    img = np.array(
        [
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
            ],
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
            ],
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
            ],
        ],
        dtype=np.uint8,
    )

    index = _voxels.determine_index_nearest_boundary(
        img=img,
        label=0,
        dimension=2,
        start=np.array([0, 0, 0], dtype=np.uint64),
        upper_skip=0,
        forward=True,
    )

    assert index == 9

    index = _voxels.determine_index_nearest_boundary(
        img=img,
        label=0,
        dimension=2,
        start=np.array([0, 0, 0], dtype=np.uint64),
        upper_skip=0,
        forward=False,
    )

    assert index == 12

    index = _voxels.determine_index_nearest_boundary(
        img=img,
        label=0,
        dimension=2,
        start=np.array([0, 0, 1], dtype=np.uint64),
        upper_skip=0,
        forward=True,
    )

    assert index == 9

    index = _voxels.determine_index_nearest_boundary(
        img=img,
        label=0,
        dimension=2,
        start=np.array([0, 0, 1], dtype=np.uint64),
        upper_skip=0,
        forward=False,
    )

    assert index == 12


@pytest.mark.figures
def test_get_nearest_boundary_index_figure(generate_padded_subdomain) -> None:
    """Test for ensuring get_nearest_boundary_index works. duh"""
    rank = 0
    sd = generate_padded_subdomain(rank)

    img = pmmoto.domain_generation.gen_img_smoothed_random_binary(sd.voxels, 0.5, 5.0)

    boundary_index = pmmoto.core.voxels.get_nearest_boundary_index(
        subdomain=sd, img=img, label=0, which_voxels="own"
    )

    img_out = np.ones_like(img)
    for feature_id, index in boundary_index.items():
        for nx in range(index.shape[0]):
            for ny in range(index.shape[1]):
                if feature_id[0] != 0:
                    if index[nx, ny] > -1:
                        img_out[int(index[nx, ny]), nx, ny] = 0
                    else:
                        img_out[0, nx, ny] = 255
                if feature_id[1] != 0:
                    if index[nx, ny] > -1:
                        img_out[nx, int(index[nx, ny]), ny] = 0
                    else:
                        img_out[nx, 0, ny] = 255
                if feature_id[2] != 0:
                    if index[nx, ny] > -1:
                        img_out[nx, ny, int(index[nx, ny])] = 0
                    else:
                        img_out[nx, ny, 0] = 255
