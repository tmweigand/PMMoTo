"""test_subdomain.py"""

import numpy as np

import pmmoto


def test_subdomain(generate_subdomain):
    """
    Test for subdomain
    """

    rank = 12
    sd = generate_subdomain(rank)

    index = sd.get_index()
    assert index == (1, 1, 0)

    sd_voxels = sd.get_voxels()
    assert sd_voxels == (33, 33, 33)

    box = sd.get_box()

    assert box == (
        ((84.59, 92.18), (3.249299999999991, 51.49859999999998), (-9.0, -7.0667247))
    )

    global_boundary = sd.get_global_boundary()
    sd_boundary_types = sd.get_boundary_types(global_boundary)

    for feature_id, is_global_boundary in global_boundary.items():
        if feature_id == (0, 0, -1):
            assert is_global_boundary
        else:
            assert not is_global_boundary

    for feature_id, boundary_type in sd_boundary_types.items():
        if feature_id == (0, 0, -1):
            assert boundary_type == "periodic"
        else:
            assert boundary_type == "internal"

    sd_inlet = sd.get_inlet()
    np.testing.assert_array_equal(sd_inlet, [0, 0, 0, 0, 0, 0])

    sd_outlet = sd.get_outlet()
    np.testing.assert_array_equal(sd_outlet, [0, 0, 0, 0, 0, 0])

    start = sd.get_start()
    assert start == (33, 33, 0)


def test_subdomain_2(generate_subdomain):
    """
    Test for subdomain
    """

    rank = 26
    sd = generate_subdomain(rank)

    index = sd.get_index()
    assert index == (2, 2, 2)

    sd_voxels = sd.get_voxels()
    assert sd_voxels == (34, 34, 34)

    box = sd.get_box()
    assert box == ((92.18, 100.0), (51.4986, 101.21), (-5.1334494, -3.14159))

    global_boundary = sd.get_global_boundary()
    sd_boundary_types = sd.get_boundary_types(global_boundary)

    global_features = {
        (1, 0, 0): "end",
        (0, 1, 0): "wall",
        (0, 0, 1): "periodic",
        (1, 0, 1): "end",
        (1, 1, 0): "end",
        (0, 1, 1): "wall",
        (1, 1, 1): "end",
    }

    for feature_id, is_global_boundary in global_boundary.items():
        if feature_id in global_features:
            assert is_global_boundary
        else:
            assert not is_global_boundary

    for feature_id, boundary_type in sd_boundary_types.items():
        if feature_id in global_features:
            assert boundary_type == global_features[feature_id]
        else:
            assert boundary_type == "internal"

    sd_inlet = sd.get_inlet()
    np.testing.assert_array_equal(sd_inlet, [0, 0, 0, 0, 0, 0])

    sd_outlet = sd.get_outlet()
    np.testing.assert_array_equal(sd_outlet, [0, 1, 0, 0, 0, 0])

    start = sd.get_start()
    assert start == (66, 66, 66)


def test_subdomain_3(generate_subdomain):
    """
    Test for subdomain
    """
    rank = 0
    sd = generate_subdomain(rank)

    index = sd.get_index()
    assert index == (0, 0, 0)

    sd_voxels = sd.get_voxels()
    assert sd_voxels == (33, 33, 33)

    box = sd.get_box()
    assert box == ((77.0, 84.59), (-45.0, 3.249299999999991), (-9.0, -7.0667247))

    global_boundary = sd.get_global_boundary()
    sd_boundary_types = sd.get_boundary_types(global_boundary)

    global_features = {
        (-1, 0, 0): "end",
        (0, -1, 0): "wall",
        (0, 0, -1): "periodic",
        (-1, 0, -1): "end",
        (-1, -1, 0): "end",
        (0, -1, -1): "wall",
        (-1, -1, -1): "end",
    }

    for feature_id, is_global_boundary in global_boundary.items():
        if feature_id in global_features:
            assert is_global_boundary
        else:
            assert not is_global_boundary

    for feature_id, boundary_type in sd_boundary_types.items():
        if feature_id in global_features:
            assert boundary_type == global_features[feature_id]
        else:
            assert boundary_type == "internal"

    sd_inlet = sd.get_inlet()
    np.testing.assert_array_equal(sd_inlet, [1, 0, 0, 0, 0, 0])

    sd_outlet = sd.get_outlet()
    np.testing.assert_array_equal(sd_outlet, [0, 0, 0, 0, 0, 0])

    start = sd.get_start()
    assert start == (0, 0, 0)
