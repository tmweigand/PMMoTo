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

    assert sd.index == (1, 1, 0)

    np.testing.assert_array_equal(sd.pad, ((1, 1), (1, 1), (1, 1)))

    assert sd.voxels == (35, 35, 35)

    assert sd.box == (
        (84.36, 92.41000000000001),
        (1.7871999999999912, 52.96069999999998),
        (-9.0585841, -7.0081406),
    )

    for feature_id, is_global_boundary in sd.global_boundary.items():
        if feature_id == (0, 0, -1):
            assert is_global_boundary
        else:
            assert not is_global_boundary

    for feature_id, boundary_type in sd.boundary_types.items():
        if feature_id == (0, 0, -1):
            assert boundary_type == "periodic"
        else:
            assert boundary_type == "internal"

    np.testing.assert_array_equal(sd.inlet, [0, 0, 0, 0, 0, 0])

    np.testing.assert_array_equal(sd.outlet, [0, 0, 0, 0, 0, 0])

    assert sd.start == (32, 32, -1)

    res_padding = sd.get_reservoir_padding(reservoir_voxels)
    np.testing.assert_array_equal(res_padding, [[0, 0], [0, 0], [0, 0]])

    own_voxels = sd.get_own_voxels(sd.pad, sd.start, sd.voxels)
    np.testing.assert_array_equal(own_voxels, [33, 68, 33, 68, 0, 35])


def test_subdomain_2():
    """
    Test for subdomain
    """

    rank = 26
    pad = (2, 2, 2)
    reservoir_voxels = 3
    sd = generate_padded_subdomain(rank, pad, reservoir_voxels)

    assert sd.index == (2, 2, 2)

    np.testing.assert_array_equal(sd.pad, ((2, 0), (2, 1), (2, 2)))

    assert sd.voxels == (36, 37, 38)

    # Test updating img
    sd_pad, _ = sd.extend_padding(pad)
    np.testing.assert_array_equal(sd_pad, ((2, 0), (2, 0), (2, 2)))

    assert sd.box == (
        (91.72000000000001, 100.0),
        (48.574400000000004, 102.6721),
        (-5.2506176, -3.0244218),
    )

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
    #     if feature_id in global_features or sd.neighbor_ranks[feature_id] < 0:
    #         assert boundary_type == global_features[feature_id]
    #     else:
    #         assert boundary_type == "internal"

    np.testing.assert_array_equal(sd.inlet, [0, 0, 0, 0, 0, 0])

    np.testing.assert_array_equal(sd.outlet, [0, 1, 0, 0, 0, 0])

    assert sd.start == (64, 64, 64)

    res_padding = sd.get_reservoir_padding(reservoir_voxels)
    np.testing.assert_array_equal(res_padding, [[0, 0], [0, 0], [0, 0]])

    own_voxels = sd.get_own_voxels(sd.pad, sd.start, sd.voxels)
    np.testing.assert_array_equal(own_voxels, [66, 102, 66, 103, 66, 104])


def test_subdomain_3():
    """
    Test for subdomain
    """

    rank = 0
    pad = (1, 1, 1)
    reservoir_voxels = 3
    sd = generate_padded_subdomain(rank, pad, reservoir_voxels)

    assert sd.index == (0, 0, 0)

    np.testing.assert_array_equal(sd.pad, ((0, 1), (1, 1), (1, 1)))
    sd_pad, extended_loop = sd.extend_padding(pad)
    np.testing.assert_array_equal(sd_pad, ((0, 1), (0, 1), (1, 1)))

    assert sd.voxels == (37, 35, 35)

    assert sd.box == (
        (76.31, 84.82000000000001),
        (-46.4621, 4.7113999999999905),
        (-9.0585841, -7.0081406),
    )

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

    # for feature_id, boundary_type in sd_boundary_types.items():
    #     if feature_id in global_features:
    #         assert boundary_type == global_features[feature_id]
    #     else:
    #         assert boundary_type == "internal"

    np.testing.assert_array_equal(sd.inlet, [1, 0, 0, 0, 0, 0])

    np.testing.assert_array_equal(sd.outlet, [0, 0, 0, 0, 0, 0])

    assert sd.start == (-3, -1, -1)

    res_padding = sd.get_reservoir_padding(reservoir_voxels)
    np.testing.assert_array_equal(res_padding, [[3, 0], [0, 0], [0, 0]])

    own_voxels = sd.get_own_voxels(sd.pad, sd.start, sd.voxels)
    np.testing.assert_array_equal(own_voxels, [-3, 34, 0, 35, 0, 35])
