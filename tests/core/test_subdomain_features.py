"""test_subdomain_features.py"""

import pmmoto
import pytest
import numpy as np


def generate_padded_subdomain(
    rank: int, pad: tuple[int, ...], reservoir_voxels: int
) -> pmmoto.core.subdomain_padded.PaddedSubdomain:
    """Generate a padded subdomain"""
    box = ((77, 100), (-45, 101.21), (-9.0, -3.14159))
    boundary_types = (
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        (pmmoto.BoundaryType.WALL, pmmoto.BoundaryType.WALL),
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
    )
    inlet = ((True, False), (False, False), (False, False))
    outlet = ((False, True), (False, False), (False, False))
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
    """Test for subdomain"""
    rank = 12
    pad = (1, 1, 1)
    reservoir_voxels = 1
    sd = generate_padded_subdomain(rank, pad, reservoir_voxels)

    sd_features = pmmoto.core.subdomain_features.SubdomainFeatures(
        sd, sd.voxels, sd.pad
    )
    features = sd_features.get_features()

    for feature_id, feature in features.items():
        if feature_id == (0, 0, -1):
            assert feature.global_boundary
        else:
            assert not feature.global_boundary

    for feature_id, feature in features.items():
        if feature_id == (0, 0, -1):
            assert feature.boundary_type == pmmoto.BoundaryType.PERIODIC
            assert (
                sd_features.get_boundary_type(feature_id)
                == pmmoto.BoundaryType.PERIODIC
            )
        else:
            assert feature.boundary_type == pmmoto.BoundaryType.INTERNAL

    periodic_features = sd_features.collect_periodic_features()
    assert periodic_features == [(0, 0, -1)]

    periodic_correction = sd_features.collect_periodic_corrections()
    assert periodic_correction == {(0, 0, -1): (0, 0, 1)}


def test_subdomain_2():
    """Test for subdomain"""
    rank = 26
    pad = (2, 2, 2)
    reservoir_voxels = 3
    sd = generate_padded_subdomain(rank, pad, reservoir_voxels)
    sd_features = pmmoto.core.subdomain_features.SubdomainFeatures(
        sd, sd.voxels, sd.pad
    )
    features = sd_features.get_features()
    global_features = {
        (1, 0, 0): "end",
        (0, 1, 0): "wall",
        (0, 0, 1): "periodic",
        (1, 0, 1): "end",
        (1, 1, 0): "end",
        (0, 1, 1): "wall",
        (1, 1, 1): "end",
    }

    for feature_id, feature in features.items():
        if feature_id in global_features:
            assert feature.global_boundary
        else:
            assert not feature.global_boundary

    for feature_id, feature in features.items():
        if feature_id in global_features:
            assert feature.boundary_type == global_features[feature_id]
        else:
            assert feature.boundary_type == "internal"

    assert sd.inlet == ((False, False), (False, False), (False, False))
    assert sd.outlet == ((False, True), (False, False), (False, False))

    assert sd.start == (64, 64, 64)

    res_padding = sd.get_reservoir_padding(reservoir_voxels)
    np.testing.assert_array_equal(res_padding, [[0, 0], [0, 0], [0, 0]])

    own_voxels = sd.get_own_voxels()
    np.testing.assert_array_equal(own_voxels, [2, 36, 2, 36, 2, 36])


def test_subdomain_3():
    """Test for subdomain"""
    rank = 0
    pad = (1, 1, 1)
    reservoir_voxels = 3
    sd = generate_padded_subdomain(rank, pad, reservoir_voxels)
    sd_features = pmmoto.core.subdomain_features.SubdomainFeatures(
        sd, sd.voxels, sd.pad
    )
    features = sd_features.get_features()

    assert sd.index == (0, 0, 0)

    np.testing.assert_array_equal(sd.pad, ((0, 1), (1, 1), (1, 1)))
    sd_pad, _ = sd.extend_padding(pad)
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

    for feature_id, feature in features.items():
        if feature_id in global_features:
            assert feature.global_boundary
        else:
            assert not feature.global_boundary

    for feature_id, feature in features.items():
        if feature_id in global_features:
            assert feature.boundary_type == global_features[feature_id]
        else:
            assert feature.boundary_type == "internal"

    assert sd.inlet == ((True, False), (False, False), (False, False))

    assert sd.outlet == ((False, False), (False, False), (False, False))
    assert sd.start == (-3, -1, -1)

    res_padding = sd.get_reservoir_padding(reservoir_voxels)
    np.testing.assert_array_equal(res_padding, [[3, 0], [0, 0], [0, 0]])

    own_voxels = sd.get_own_voxels()
    np.testing.assert_array_equal(own_voxels, [3, 36, 1, 34, 1, 34])


def test_get_feature_member():
    """Tests for get info of a specific feature"""
    rank = 0
    pad = (1, 1, 1)
    reservoir_voxels = 3
    sd = generate_padded_subdomain(rank, pad, reservoir_voxels)
    sd_features = pmmoto.core.subdomain_features.SubdomainFeatures(
        sd, sd.voxels, sd.pad
    )

    feature_member = sd_features.get_feature_member((1, 0, 0), "feature_id")
    assert feature_member == (1, 0, 0)

    feature_member = sd_features.get_feature_member((1, 0, 1), "feature_id")
    assert feature_member == (1, 0, 1)

    feature_member = sd_features.get_feature_member((1, -1, 1), "feature_id")
    assert feature_member == (1, -1, 1)

    with pytest.raises(KeyError):
        sd_features.get_feature_member((1, -2, 1), "error")

    with pytest.raises(AttributeError):
        sd_features.get_feature_member((1, -1, 1), "error")
