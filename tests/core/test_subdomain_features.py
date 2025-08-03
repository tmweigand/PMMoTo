"""test_subdomain_features.py"""

import numpy as np
import pmmoto


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
        else:
            if isinstance(feature, pmmoto.core.features.Face):
                assert feature.boundary_type == pmmoto.BoundaryType.INTERNAL
            elif isinstance(feature, pmmoto.core.features.Edge):
                if feature_id[2] == -1:
                    assert feature.boundary_type == (
                        pmmoto.BoundaryType.INTERNAL,
                        pmmoto.BoundaryType.PERIODIC,
                    )
                else:
                    assert feature.boundary_type == (
                        pmmoto.BoundaryType.INTERNAL,
                        pmmoto.BoundaryType.INTERNAL,
                    )
            elif isinstance(feature, pmmoto.core.features.Corner):
                if feature_id[2] == -1:
                    assert feature.boundary_type == (
                        pmmoto.BoundaryType.INTERNAL,
                        pmmoto.BoundaryType.INTERNAL,
                        pmmoto.BoundaryType.PERIODIC,
                    )
                else:
                    assert feature.boundary_type == (
                        pmmoto.BoundaryType.INTERNAL,
                        pmmoto.BoundaryType.INTERNAL,
                        pmmoto.BoundaryType.INTERNAL,
                    )


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
        (1, 0, 0): pmmoto.BoundaryType.END,
        (0, 1, 0): pmmoto.BoundaryType.WALL,
        (0, 0, 1): pmmoto.BoundaryType.PERIODIC,
        (1, 0, 1): (pmmoto.BoundaryType.END, pmmoto.BoundaryType.PERIODIC),
        (1, 1, 0): (pmmoto.BoundaryType.END, pmmoto.BoundaryType.WALL),
        (0, 1, 1): (pmmoto.BoundaryType.WALL, pmmoto.BoundaryType.PERIODIC),
        (1, 1, 1): (
            pmmoto.BoundaryType.END,
            pmmoto.BoundaryType.WALL,
            pmmoto.BoundaryType.PERIODIC,
        ),
    }

    for feature_id, feature in features.items():
        if feature_id in global_features:
            assert feature.global_boundary
        else:
            assert not feature.global_boundary

    for feature_id, feature in features.items():
        if feature_id in global_features:
            assert feature.boundary_type == global_features[feature_id]

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
        (-1, 0, 0): pmmoto.BoundaryType.END,
        (0, -1, 0): pmmoto.BoundaryType.WALL,
        (0, 0, -1): pmmoto.BoundaryType.PERIODIC,
        (-1, 0, -1): (pmmoto.BoundaryType.END, pmmoto.BoundaryType.PERIODIC),
        (-1, -1, 0): (pmmoto.BoundaryType.END, pmmoto.BoundaryType.WALL),
        (0, -1, -1): (pmmoto.BoundaryType.WALL, pmmoto.BoundaryType.PERIODIC),
        (-1, -1, -1): (
            pmmoto.BoundaryType.END,
            pmmoto.BoundaryType.WALL,
            pmmoto.BoundaryType.PERIODIC,
        ),
    }

    for feature_id, feature in features.items():
        if feature_id in global_features:
            assert feature.global_boundary
        else:
            assert not feature.global_boundary

    for feature_id, feature in features.items():
        if feature_id in global_features:
            assert feature.boundary_type == global_features[feature_id]

    assert sd.inlet == ((True, False), (False, False), (False, False))

    assert sd.outlet == ((False, False), (False, False), (False, False))
    assert sd.start == (-3, -1, -1)

    res_padding = sd.get_reservoir_padding(reservoir_voxels)
    np.testing.assert_array_equal(res_padding, [[3, 0], [0, 0], [0, 0]])

    own_voxels = sd.get_own_voxels()
    np.testing.assert_array_equal(own_voxels, [3, 36, 1, 34, 1, 34])


# def test_collect_features() -> None:
#     """Test for subdomain features"""
#     rank = 0
#     pad = (1, 1, 1)
#     reservoir_voxels = 0
#     sd = generate_padded_subdomain(rank, pad, reservoir_voxels)

#     features = pmmoto.core.subdomain_features.SubdomainFeatures(sd, sd.voxels, sd.pad)

#     print(features)

#     # assert len(features["faces"]) == 6
#     # assert len(features["edges"]) == 12
#     # assert len(features["corners"]) == 8


# def test_feature_voxels_face():
#     """Test get_feature_voxels"""
#     feature_id = (-1, 0, 0)
#     voxels = (10, 10, 10)
#     pad = [[0, 0], [0, 0], [0, 0]]

#     feature_voxels = pmmoto.core.subdomain_features.get_feature_voxels(
#         feature_id=feature_id, voxels=voxels, pad=pad
#     )

#     np.testing.assert_array_equal(
#         feature_voxels["own"], np.array([[0, 1], [0, 10], [0, 10]])
#     )

#     feature_id = (-1, 0, 0)
#     voxels = (10, 10, 10)
#     pad = [[1, 1], [1, 1], [1, 1]]

#     feature_voxels = pmmoto.core.subdomain_features.get_feature_voxels(
#         feature_id=feature_id, voxels=voxels, pad=pad
#     )

#     np.testing.assert_array_equal(
#         feature_voxels["own"], np.array([[1, 2], [1, 9], [1, 9]])
#     )

#     np.testing.assert_array_equal(
#         feature_voxels["neighbor"], np.array([[0, 1], [1, 9], [1, 9]])
#     )

#     feature_id = (-1, 0, 0)
#     voxels = (10, 10, 10)
#     pad = [[4, 4], [2, 2], [3, 3]]

#     feature_voxels = pmmoto.core.subdomain_features.get_feature_voxels(
#         feature_id=feature_id, voxels=voxels, pad=pad
#     )

#     print(feature_voxels["own"])
#     print(feature_voxels["neighbor"])

#     # np.testing.assert_array_equal(
#     #     feature_voxels["own"], np.array([[1, 2], [1, 9], [1, 9]])
#     # )

#     # np.testing.assert_array_equal(
#     #     feature_voxels["neighbor"], np.array([[0, 1], [1, 9], [1, 9]])
#     # )


# def test_feature_voxels_edge():
#     """Test get_feature_voxels"""
#     feature_id = (-1, 0, 1)
#     voxels = (10, 10, 10)
#     pad = [[0, 0], [0, 0], [0, 0]]

#     feature_voxels = pmmoto.core.subdomain_features.get_feature_voxels(
#         feature_id=feature_id, voxels=voxels, pad=pad
#     )

#     np.testing.assert_array_equal(
#         feature_voxels["own"], np.array([[0, 1], [0, 10], [9, 10]])
#     )

#     feature_id = (-1, 0, 1)
#     voxels = (10, 10, 10)
#     pad = [[1, 1], [1, 1], [1, 1]]

#     feature_voxels = pmmoto.core.subdomain_features.get_feature_voxels(
#         feature_id=feature_id, voxels=voxels, pad=pad
#     )

#     np.testing.assert_array_equal(
#         feature_voxels["own"], np.array([[1, 2], [1, 9], [8, 9]])
#     )

#     np.testing.assert_array_equal(
#         feature_voxels["neighbor"], np.array([[0, 1], [1, 9], [9, 10]])
#     )


# def test_feature_voxels_corner():
#     """Test get_feature_voxels"""
#     feature_id = (1, 1, -1)
#     voxels = (10, 10, 10)
#     pad = [[0, 0], [0, 0], [0, 0]]

#     feature_voxels = pmmoto.core.subdomain_features.get_feature_voxels(
#         feature_id=feature_id, voxels=voxels, pad=pad
#     )

#     np.testing.assert_array_equal(
#         feature_voxels["own"], np.array([[9, 10], [9, 10], [0, 1]])
#     )

#     feature_id = (1, 1, -1)
#     voxels = (10, 10, 10)
#     pad = [[1, 1], [1, 1], [1, 1]]

#     feature_voxels = pmmoto.core.subdomain_features.get_feature_voxels(
#         feature_id=feature_id, voxels=voxels, pad=pad
#     )

#     np.testing.assert_array_equal(
#         feature_voxels["own"], np.array([[8, 9], [8, 9], [1, 2]])
#     )

#     np.testing.assert_array_equal(
#         feature_voxels["neighbor"], np.array([[9, 10], [9, 10], [0, 1]])
#     )


# def test_collect_periodic_features():
#     """Check function that loops through the subdomain features

#     Returns a list of all of the periodic ones.
#     """
#     rank = 26
#     pad = (1, 1, 1)
#     reservoir_voxels = 0
#     sd = generate_padded_subdomain(rank, pad, reservoir_voxels)

#     periodic_features = pmmoto.core.subdomain_features.collect_periodic_features(
#         sd.features
#     )

#     np.testing.assert_equal(periodic_features, [(0, 0, 1)])


# def test_collect_periodic_corrections():
#     """Check function that loops through the subdomain features.

#     Returns a list of all of the periodic corrections.
#     """
#     rank = 26
#     pad = (1, 1, 1)
#     reservoir_voxels = 0
#     sd = generate_padded_subdomain(rank, pad, reservoir_voxels)

#     periodic_corrections = pmmoto.core.subdomain_features.collect_periodic_corrections(
#         sd.features
#     )

#     np.testing.assert_equal(
#         periodic_corrections,
#         {
#             (0, 0, 1): (0, 0, -1),
#         },
#     )
