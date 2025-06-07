"""test_subdomain_padded.py"""

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


def test_subdomain() -> None:
    """Test for subdomain"""
    rank = 12
    pad = (1, 1, 1)
    reservoir_voxels = 1
    sd = generate_padded_subdomain(rank, pad, reservoir_voxels)

    assert sd.index == (1, 1, 0)

    np.testing.assert_array_equal(sd.pad, ((1, 1), (1, 1), (1, 1)))

    assert sd.voxels == (35, 35, 35)
    assert sd.own_voxels == (33, 33, 33)

    assert sd.box == (
        (84.36, 92.41000000000001),
        (1.7871999999999912, 52.96069999999998),
        (-9.0585841, -7.0081406),
    )

    np.testing.assert_array_equal(
        sd.global_boundary, ((False, False), (False, False), (True, False))
    )

    assert sd.boundary_types == (
        (pmmoto.BoundaryType.INTERNAL, pmmoto.BoundaryType.INTERNAL),
        (pmmoto.BoundaryType.INTERNAL, pmmoto.BoundaryType.INTERNAL),
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.INTERNAL),
    )

    np.testing.assert_array_equal(
        sd.inlet, [[False, False], [False, False], [False, False]]
    )

    np.testing.assert_array_equal(
        sd.outlet, [[False, False], [False, False], [False, False]]
    )

    assert sd.start == (32, 32, -1)

    res_padding = sd.get_reservoir_padding(reservoir_voxels)
    np.testing.assert_array_equal(res_padding, [[0, 0], [0, 0], [0, 0]])

    own_voxels = sd.get_own_voxels()
    np.testing.assert_array_equal(own_voxels, [1, 34, 1, 34, 1, 34])


def test_subdomain_2() -> None:
    """Test for subdomain"""
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

    np.testing.assert_array_equal(
        sd.global_boundary, ((False, True), (False, True), (False, True))
    )

    assert sd.boundary_types == (
        (pmmoto.BoundaryType.INTERNAL, pmmoto.BoundaryType.END),
        (pmmoto.BoundaryType.INTERNAL, pmmoto.BoundaryType.WALL),
        (pmmoto.BoundaryType.INTERNAL, pmmoto.BoundaryType.PERIODIC),
    )

    np.testing.assert_array_equal(
        sd.inlet, ((False, False), (False, False), (False, False))
    )

    np.testing.assert_array_equal(
        sd.outlet, ((False, True), (False, False), (False, False))
    )

    assert sd.start == (64, 64, 64)

    res_padding = sd.get_reservoir_padding(reservoir_voxels)
    np.testing.assert_array_equal(res_padding, [[0, 0], [0, 0], [0, 0]])

    own_voxels = sd.get_own_voxels()
    np.testing.assert_array_equal(own_voxels, [2, 36, 2, 36, 2, 36])


def test_subdomain_3() -> None:
    """Test for subdomain"""
    rank = 0
    pad = (1, 1, 1)
    reservoir_voxels = 3
    sd = generate_padded_subdomain(rank, pad, reservoir_voxels)

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

    np.testing.assert_array_equal(
        sd.global_boundary, ((True, False), (True, False), (True, False))
    )

    assert sd.boundary_types == (
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.INTERNAL),
        (pmmoto.BoundaryType.WALL, pmmoto.BoundaryType.INTERNAL),
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.INTERNAL),
    )

    np.testing.assert_array_equal(
        sd.inlet, ((True, False), (False, False), (False, False))
    )

    np.testing.assert_array_equal(
        sd.outlet, ((False, False), (False, False), (False, False))
    )

    assert sd.start == (-3, -1, -1)

    res_padding = sd.get_reservoir_padding(reservoir_voxels)
    np.testing.assert_array_equal(res_padding, [[3, 0], [0, 0], [0, 0]])

    own_voxels = sd.get_own_voxels()
    np.testing.assert_array_equal(own_voxels, [3, 36, 1, 34, 1, 34])


def test_own_voxels() -> None:
    """Ensures that walls are correctly added to a porous media img"""
    boundary_types = (
        (pmmoto.BoundaryType.WALL, pmmoto.BoundaryType.WALL),
        (pmmoto.BoundaryType.WALL, pmmoto.BoundaryType.WALL),
        (pmmoto.BoundaryType.WALL, pmmoto.BoundaryType.WALL),
    )
    sd = pmmoto.initialize(
        voxels=(10, 10, 10),
        boundary_types=boundary_types,
    )
    np.testing.assert_array_equal(sd.get_own_voxels(), [1, 11, 1, 11, 1, 11])

    # # Boundary Conditions must be 0 for reservoir check
    inlet = ((True, False), (False, False), (False, False))
    sd = pmmoto.initialize(
        voxels=(10, 10, 10),
        boundary_types=boundary_types,
        inlet=inlet,
        reservoir_voxels=10,
    )
    np.testing.assert_array_equal(sd.get_own_voxels(), [1, 11, 1, 11, 1, 11])

    # Now test reservoir
    inlet = ((True, False), (False, False), (False, False))
    boundary_types = (
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
    )
    sd = pmmoto.initialize(
        voxels=(10, 10, 10),
        boundary_types=boundary_types,
        reservoir_voxels=10,
        inlet=inlet,
    )
    np.testing.assert_array_equal(sd.get_own_voxels(), [10, 20, 0, 10, 0, 10])
