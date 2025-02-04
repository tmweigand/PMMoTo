"""test_subdomain_padded.py"""

import numpy as np
import pmmoto


def generate_padded_subdomain(rank, pad, reservoir_voxels):
    """
    Generate a padded subdomain
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


def test_subdomain():
    """
    Test for subdomain
    """

    rank = 12
    pad = (1, 1, 1)
    reservoir_voxels = 1
    sd = generate_padded_subdomain(rank, pad, reservoir_voxels)

    index = sd.get_index()
    assert index == (1, 1, 0)

    sd_pad = sd.get_padding(pad)
    np.testing.assert_array_equal(sd_pad, ((1, 1), (1, 1), (1, 1)))

    sd_voxels = sd.get_voxels()
    assert sd_voxels == (35, 35, 35)

    box = sd.get_box()

    assert box == (
        (84.36, 92.41000000000001),
        (1.7871999999999912, 52.96069999999998),
        (-9.0585841, -7.0081406),
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
    assert start == (32, 32, -1)

    res_padding = sd.get_reservoir_padding(reservoir_voxels)
    np.testing.assert_array_equal(res_padding, [[0, 0], [0, 0], [0, 0]])

    own_voxels = sd.get_own_voxels(sd_pad, start, sd_voxels)
    np.testing.assert_array_equal(own_voxels, [33, 68, 33, 68, 0, 35])


def test_subdomain_2():
    """
    Test for subdomain
    """

    rank = 26
    pad = (2, 2, 2)
    reservoir_voxels = 3
    sd = generate_padded_subdomain(rank, pad, reservoir_voxels)

    index = sd.get_index()
    assert index == (2, 2, 2)

    sd_pad = sd.get_padding(pad)
    np.testing.assert_array_equal(sd_pad, ((2, 0), (2, 1), (2, 2)))

    sd_voxels = sd.get_voxels()
    assert sd_voxels == (36, 38, 38)

    box = sd.get_box()

    assert box == (
        (91.72000000000001, 100.0),
        (48.574400000000004, 104.13419999999999),
        (-5.2506176, -3.0244218),
    )

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
    assert start == (64, 64, 64)

    res_padding = sd.get_reservoir_padding(reservoir_voxels)
    np.testing.assert_array_equal(res_padding, [[0, 0], [0, 0], [0, 0]])

    own_voxels = sd.get_own_voxels(sd_pad, start, sd_voxels)
    np.testing.assert_array_equal(own_voxels, [66, 102, 66, 104, 66, 104])


def test_subdomain_3():
    """
    Test for subdomain
    """

    rank = 0
    pad = (1, 1, 1)
    reservoir_voxels = 3
    sd = generate_padded_subdomain(rank, pad, reservoir_voxels)

    index = sd.get_index()
    assert index == (0, 0, 0)

    sd_pad = sd.get_padding(pad)
    np.testing.assert_array_equal(sd_pad, ((0, 1), (1, 1), (1, 1)))

    sd_voxels = sd.get_voxels()
    assert sd_voxels == (37, 35, 35)

    box = sd.get_box()

    assert box == (
        (76.31, 84.82000000000001),
        (-46.4621, 4.7113999999999905),
        (-9.0585841, -7.0081406),
    )

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
    assert start == (-3, -1, -1)

    res_padding = sd.get_reservoir_padding(reservoir_voxels)
    np.testing.assert_array_equal(res_padding, [[3, 0], [0, 0], [0, 0]])

    own_voxels = sd.get_own_voxels(sd_pad, start, sd_voxels)
    np.testing.assert_array_equal(own_voxels, [-3, 34, 0, 35, 0, 35])
