"""test_subdomain.py"""

import pytest
import numpy as np

import pmmoto


def test_subdomain(generate_subdomain):
    """Test for subdomain
    """
    rank = 12
    sd = generate_subdomain(rank)

    assert sd.index == (1, 1, 0)

    assert sd.voxels == (33, 33, 33)

    assert sd.box == (
        ((84.59, 92.18), (3.249299999999991, 51.49859999999998), (-9.0, -7.0667247))
    )

    for feature_id, is_global_boundary in sd.global_boundary.items():
        if feature_id == (0, 0, -1):
            assert is_global_boundary
        else:
            assert not is_global_boundary

    # for feature_id, boundary_type in sd.boundary_types.items():
    #     if feature_id == (0, 0, -1):
    #         assert boundary_type == "periodic"
    #     else:
    #         assert boundary_type == "internal"

    np.testing.assert_array_equal(sd.inlet, [0, 0, 0, 0, 0, 0])

    np.testing.assert_array_equal(sd.outlet, [0, 0, 0, 0, 0, 0])

    assert sd.start == (33, 33, 0)


def test_subdomain_2(generate_subdomain):
    """Test for subdomain
    """
    rank = 26
    sd = generate_subdomain(rank)

    assert sd.index == (2, 2, 2)

    assert sd.voxels == (34, 34, 34)

    assert sd.box == ((92.18, 100.0), (51.4986, 101.21), (-5.1334494, -3.14159))

    global_features = {
        (1, 0, 0): "end",
        (0, 1, 0): "wall",
        (0, 0, 1): "periodic",
        (1, 0, 1): "end",
        (1, 1, 0): "end",
        (0, 1, 1): "wall",
        (1, 1, 1): "end",
    }

    for feature_id, is_global_boundary in sd.global_boundary.items():
        if feature_id in global_features:
            assert is_global_boundary
        else:
            assert not is_global_boundary

    # for feature_id, boundary_type in sd.boundary_types.items():
    #     if feature_id in global_features:
    #         assert boundary_type == global_features[feature_id]
    #     else:
    #         assert boundary_type == "internal"

    np.testing.assert_array_equal(sd.inlet, [0, 0, 0, 0, 0, 0])

    np.testing.assert_array_equal(sd.outlet, [0, 1, 0, 0, 0, 0])

    assert sd.start == (66, 66, 66)


def test_subdomain_3(generate_subdomain):
    """Test for subdomain
    """
    rank = 0
    sd = generate_subdomain(rank)

    assert sd.index == (0, 0, 0)

    assert sd.voxels == (33, 33, 33)

    assert sd.box == ((77.0, 84.59), (-45.0, 3.249299999999991), (-9.0, -7.0667247))

    global_features = {
        (-1, 0, 0): "end",
        (0, -1, 0): "wall",
        (0, 0, -1): "periodic",
        (-1, 0, -1): "end",
        (-1, -1, 0): "end",
        (0, -1, -1): "wall",
        (-1, -1, -1): "end",
    }

    for feature_id, is_global_boundary in sd.global_boundary.items():
        if feature_id in global_features:
            assert is_global_boundary
        else:
            assert not is_global_boundary

    # for feature_id, boundary_type in sd.boundary_types.items():
    #     if feature_id in global_features:
    #         assert boundary_type == global_features[feature_id]
    #     else:
    #         assert boundary_type == "internal"

    np.testing.assert_array_equal(sd.inlet, [1, 0, 0, 0, 0, 0])

    np.testing.assert_array_equal(sd.outlet, [0, 0, 0, 0, 0, 0])

    assert sd.start == (0, 0, 0)


@pytest.mark.figures
def test_subdomain_figures(generate_subdomain):
    """Generate images for testing subdomain
    """
    rank = 0
    sd = pmmoto.initialize(
        voxels=(100, 100, 100),
        boundary_types=((1, 1), (1, 1), (2, 2)),
        rank=rank,
        subdomains=(2, 2, 2),
        pad=(2, 2, 2),
    )

    img = np.zeros(sd.voxels)

    kind = "neighbor"
    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in sd.features[feature_type].items():
            print(feature_id, feature.neighbor_rank, feature.boundary_type)
            if feature.neighbor_rank > -1:
                img[
                    feature.loop[kind][0][0] : feature.loop[kind][0][1],
                    feature.loop[kind][1][0] : feature.loop[kind][1][1],
                    feature.loop[kind][2][0] : feature.loop[kind][2][1],
                ] = 1

    pmmoto.io.output.save_img_data_proc("data_out/test_subdomain", sd, img)


def test_walls():
    """Ensures that walls are correctly added to a porous media img
    """
    sd = pmmoto.initialize(voxels=(10, 10, 10), boundary_types=((1, 1), (1, 1), (1, 1)))

    img = np.ones(sd.voxels)
    img = sd.set_wall_bcs(img)

    assert np.all(img[0, :, :] == 0)
    assert np.all(img[-1, :, :] == 0)
    assert np.all(img[:, 0, :] == 0)
    assert np.all(img[:, -1, :] == 0)
    assert np.all(img[:, :, 0] == 0)
    assert np.all(img[:, :, -1] == 0)

    assert not np.all(img[1, :, :] == 0)
    assert not np.all(img[-2, :, :] == 0)
    assert not np.all(img[:, 1, :] == 0)
    assert not np.all(img[:, -2, :] == 0)
    assert not np.all(img[:, :, 1] == 0)
    assert not np.all(img[:, :, -2] == 0)


def test_get_img_index():
    """Ensure the correct index is provided given physical coordinates
    """
    sd = pmmoto.initialize((10, 10, 10))

    assert sd.get_img_index((0.5, 0.5, 0.5)) == (5, 5, 5)
    assert sd.get_img_index((0.49, 0.49, 0.49)) == (4, 4, 4)
    assert sd.get_img_index((0.01, 0.99, 0.33)) == (0, 9, 3)
