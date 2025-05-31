"""test_subdomain_features.py"""

import pytest
import numpy as np
import pmmoto


def generate_padded_subdomain(rank, pad, reservoir_voxels):
    """Generate a padded subdomain
    """
    box = ((77, 100), (-45, 101.21), (-9.0, -3.14159))
    boundary_types = ((0, 0), (1, 1), (2, 2))
    inlet = ((1, 0), (0, 0), (0, 0))
    outlet = ((0, 1), (0, 0), (0, 0))
    voxels = (100, 100, 100)
    subdomains = (3, 3, 3)

    sd = pmmoto.initialize(
        box=box,
        subdomains=subdomains,
        voxels=voxels,
        boundary_types=boundary_types,
        inlet=inlet,
        outlet=outlet,
        reservoir_voxels=reservoir_voxels,
        rank=rank,
        pad=pad,
    )

    return sd


def test_collect_features():
    """Test for subdomain features
    """
    rank = 0
    pad = (1, 1, 1)
    reservoir_voxels = 0
    sd = generate_padded_subdomain(rank, pad, reservoir_voxels)

    features = pmmoto.core.subdomain_features.collect_features(
        neighbor_ranks=sd.neighbor_ranks,
        global_boundary=sd.global_boundary,
        boundary_types=sd.boundary_types,
        voxels=sd.voxels,
    )

    assert len(features["faces"]) == 6
    assert len(features["edges"]) == 12
    assert len(features["corners"]) == 8


def test_feature_voxels_face():
    """Test get_feature_voxels
    """
    feature_id = (-1, 0, 0)
    voxels = (10, 10, 10)
    pad = [[0, 0], [0, 0], [0, 0]]

    feature_voxels = pmmoto.core.subdomain_features.get_feature_voxels(
        feature_id=feature_id, voxels=voxels, pad=pad
    )

    np.testing.assert_array_equal(
        feature_voxels["own"], np.array([[0, 1], [0, 10], [0, 10]])
    )

    feature_id = (-1, 0, 0)
    voxels = (10, 10, 10)
    pad = [[1, 1], [1, 1], [1, 1]]

    feature_voxels = pmmoto.core.subdomain_features.get_feature_voxels(
        feature_id=feature_id, voxels=voxels, pad=pad
    )

    np.testing.assert_array_equal(
        feature_voxels["own"], np.array([[1, 2], [1, 9], [1, 9]])
    )

    np.testing.assert_array_equal(
        feature_voxels["neighbor"], np.array([[0, 1], [1, 9], [1, 9]])
    )

    feature_id = (-1, 0, 0)
    voxels = (10, 10, 10)
    pad = [[4, 4], [2, 2], [3, 3]]

    feature_voxels = pmmoto.core.subdomain_features.get_feature_voxels(
        feature_id=feature_id, voxels=voxels, pad=pad
    )

    print(feature_voxels["own"])
    print(feature_voxels["neighbor"])

    # np.testing.assert_array_equal(
    #     feature_voxels["own"], np.array([[1, 2], [1, 9], [1, 9]])
    # )

    # np.testing.assert_array_equal(
    #     feature_voxels["neighbor"], np.array([[0, 1], [1, 9], [1, 9]])
    # )


def test_feature_voxels_edge():
    """Test get_feature_voxels
    """
    feature_id = (-1, 0, 1)
    voxels = (10, 10, 10)
    pad = [[0, 0], [0, 0], [0, 0]]

    feature_voxels = pmmoto.core.subdomain_features.get_feature_voxels(
        feature_id=feature_id, voxels=voxels, pad=pad
    )

    np.testing.assert_array_equal(
        feature_voxels["own"], np.array([[0, 1], [0, 10], [9, 10]])
    )

    feature_id = (-1, 0, 1)
    voxels = (10, 10, 10)
    pad = [[1, 1], [1, 1], [1, 1]]

    feature_voxels = pmmoto.core.subdomain_features.get_feature_voxels(
        feature_id=feature_id, voxels=voxels, pad=pad
    )

    np.testing.assert_array_equal(
        feature_voxels["own"], np.array([[1, 2], [1, 9], [8, 9]])
    )

    np.testing.assert_array_equal(
        feature_voxels["neighbor"], np.array([[0, 1], [1, 9], [9, 10]])
    )


def test_feature_voxels_corner():
    """Test get_feature_voxels
    """
    feature_id = (1, 1, -1)
    voxels = (10, 10, 10)
    pad = [[0, 0], [0, 0], [0, 0]]

    feature_voxels = pmmoto.core.subdomain_features.get_feature_voxels(
        feature_id=feature_id, voxels=voxels, pad=pad
    )

    np.testing.assert_array_equal(
        feature_voxels["own"], np.array([[9, 10], [9, 10], [0, 1]])
    )

    feature_id = (1, 1, -1)
    voxels = (10, 10, 10)
    pad = [[1, 1], [1, 1], [1, 1]]

    feature_voxels = pmmoto.core.subdomain_features.get_feature_voxels(
        feature_id=feature_id, voxels=voxels, pad=pad
    )

    np.testing.assert_array_equal(
        feature_voxels["own"], np.array([[8, 9], [8, 9], [1, 2]])
    )

    np.testing.assert_array_equal(
        feature_voxels["neighbor"], np.array([[9, 10], [9, 10], [0, 1]])
    )


@pytest.mark.figures
def test_feature_voxels_figure():
    """Generate output to visually inspect get_feature_voxels
    """
    voxels = (20, 20, 20)
    img = np.zeros(voxels)
    pad = [[1, 1], [1, 1], [1, 1]]
    extension = [[2, 0], [2, 0], [2, 0]]
    summed = [[3, 0], [3, 0], [3, 0]]

    feature_id = (-1, 0, 0)

    feature_voxels = pmmoto.core.subdomain_features.get_feature_voxels(
        feature_id=feature_id, voxels=voxels, pad=pad
    )

    summed = pmmoto.core.subdomain_features.get_feature_voxels(
        feature_id=feature_id, voxels=voxels, pad=summed
    )

    print("feature", feature_voxels)
    print(summed)

    updated_img = pmmoto.core.utils.constant_pad_img(img, extension, 255)

    img[
        feature_voxels["own"][0][0] : feature_voxels["own"][0][1],
        feature_voxels["own"][1][0] : feature_voxels["own"][1][1],
        feature_voxels["own"][2][0] : feature_voxels["own"][2][1],
    ] = 1

    pmmoto.io.save_img("data_out/test_feature_voxels", img)

    updated_img[
        feature_voxels["neighbor"][0][0] : feature_voxels["neighbor"][0][1],
        feature_voxels["neighbor"][1][0] : feature_voxels["neighbor"][1][1],
        feature_voxels["neighbor"][2][0] : feature_voxels["neighbor"][2][1],
    ] = 2

    pmmoto.io.save_img("data_out/test_feature_voxels_updated", updated_img)


def test_collect_periodic_features():
    """Check function that loops through the subdomain features and
    returns a list of all of the periodic ones.
    """
    rank = 26
    pad = (1, 1, 1)
    reservoir_voxels = 0
    sd = generate_padded_subdomain(rank, pad, reservoir_voxels)

    periodic_features = pmmoto.core.subdomain_features.collect_periodic_features(
        sd.features
    )

    np.testing.assert_equal(periodic_features, [(0, 0, 1)])


def test_collect_periodic_corrections():
    """Check function that loops through the subdomain features and
    returns a list of all of the periodic corrections.
    """
    rank = 26
    pad = (1, 1, 1)
    reservoir_voxels = 0
    sd = generate_padded_subdomain(rank, pad, reservoir_voxels)

    periodic_corrections = pmmoto.core.subdomain_features.collect_periodic_corrections(
        sd.features
    )

    np.testing.assert_equal(
        periodic_corrections,
        {
            (0, 0, 1): (0, 0, -1),
        },
    )
