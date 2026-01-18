"""test_lattice_packings.py"""

import pmmoto
import numpy as np


def test_base_class():
    """Test for bass class"""
    sd = pmmoto.initialize(voxels=(10, 10, 10))
    bc = pmmoto.domain_generation.lattice_packings.Lattice(sd, 1)
    np.testing.assert_array_equal(
        bc.get_basis_vectors(),
        np.array([]),
    )

    assert bc.get_radius() == 0.0


def test_simple_cubic():
    """Test for Simple Cubic Packings"""
    sd = pmmoto.initialize(voxels=(10, 10, 10))
    sc = pmmoto.domain_generation.lattice_packings.SimpleCubic(
        subdomain=sd, lattice_constant=1.0
    )

    np.testing.assert_array_equal(
        sc.generate_lattice(),
        np.array(
            [
                [0.0, 0.0, 1.0, 0.5],
                [1.0, 1.0, 0.0, 0.5],
                [0.0, 1.0, 0.0, 0.5],
                [1.0, 1.0, 1.0, 0.5],
                [0.0, 1.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 0.5],
                [1.0, 0.0, 1.0, 0.5],
                [1.0, 0.0, 0.0, 0.5],
            ]
        ),
    )

    np.testing.assert_array_equal(sc.get_basis_vectors(), np.array([[0, 0, 0]]))

    assert sc.get_coordination_number() == 6
    assert sc.get_packing_efficiency() == 52
    assert sc.get_radius() == 0.5


def test_body_centered_cubic():
    """Test for Body Centered Cubic Packings"""
    sd = pmmoto.initialize(voxels=(10, 10, 10))
    bcc = pmmoto.domain_generation.lattice_packings.BodyCenteredCubic(
        subdomain=sd, lattice_constant=1.0
    )

    np.testing.assert_array_almost_equal(
        bcc.generate_lattice(),
        np.array(
            [
                [1.5, 1.5, 0.5, 0.4330127],
                [0.0, 0.0, 1.0, 0.4330127],
                [1.0, 1.0, 0.0, 0.4330127],
                [0.5, 1.5, 0.5, 0.4330127],
                [1.5, 0.5, 0.5, 0.4330127],
                [0.5, 1.5, 1.5, 0.4330127],
                [1.0, 1.0, 1.0, 0.4330127],
                [0.0, 0.0, 0.0, 0.4330127],
                [1.5, 0.5, 1.5, 0.4330127],
                [0.0, 1.0, 0.0, 0.4330127],
                [0.0, 1.0, 1.0, 0.4330127],
                [1.0, 0.0, 1.0, 0.4330127],
                [1.5, 1.5, 1.5, 0.4330127],
                [0.5, 0.5, 0.5, 0.4330127],
                [1.0, 0.0, 0.0, 0.4330127],
                [0.5, 0.5, 1.5, 0.4330127],
            ]
        ),
    )

    np.testing.assert_array_equal(
        bcc.get_basis_vectors(), np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    )

    assert bcc.get_coordination_number() == 8.0
    assert bcc.get_packing_efficiency() == 68
    assert bcc.get_radius() == 0.4330127018922193


def test_face_centered_cubic():
    """Test for Face Centered Cubic Packings"""
    sd = pmmoto.initialize(voxels=(10, 10, 10))
    fcc = pmmoto.domain_generation.lattice_packings.FaceCenteredCubic(
        subdomain=sd, lattice_constant=1.0
    )

    np.testing.assert_array_almost_equal(
        fcc.generate_lattice(),
        np.array(
            [
                [0.5, 0.0, 1.5, 0.35355339],
                [0.0, 0.5, 0.5, 0.35355339],
                [1.5, 0.0, 0.5, 0.35355339],
                [1.0, 0.0, 1.0, 0.35355339],
                [1.5, 0.5, 1.0, 0.35355339],
                [0.0, 1.5, 1.5, 0.35355339],
                [1.5, 1.5, 1.0, 0.35355339],
                [0.5, 0.5, 0.0, 0.35355339],
                [1.0, 0.5, 0.5, 0.35355339],
                [0.5, 1.5, 1.0, 0.35355339],
                [0.0, 1.0, 1.0, 0.35355339],
                [0.5, 1.0, 0.5, 0.35355339],
                [1.0, 1.5, 0.5, 0.35355339],
                [1.5, 0.0, 1.5, 0.35355339],
                [1.0, 0.0, 0.0, 0.35355339],
                [1.5, 1.5, 0.0, 0.35355339],
                [1.0, 1.0, 0.0, 0.35355339],
                [1.5, 1.0, 1.5, 0.35355339],
                [1.5, 0.5, 0.0, 0.35355339],
                [0.5, 0.0, 0.5, 0.35355339],
                [0.0, 0.5, 1.5, 0.35355339],
                [0.0, 0.0, 1.0, 0.35355339],
                [0.5, 0.5, 1.0, 0.35355339],
                [0.5, 1.5, 0.0, 0.35355339],
                [0.5, 1.0, 1.5, 0.35355339],
                [0.0, 1.0, 0.0, 0.35355339],
                [1.5, 1.0, 0.5, 0.35355339],
                [0.0, 1.5, 0.5, 0.35355339],
                [1.0, 1.5, 1.5, 0.35355339],
                [1.0, 0.5, 1.5, 0.35355339],
                [1.0, 1.0, 1.0, 0.35355339],
                [0.0, 0.0, 0.0, 0.35355339],
            ]
        ),
    )

    np.testing.assert_array_equal(
        fcc.get_basis_vectors(),
        np.array(
            [
                [0, 0, 0],
                [0.5, 0.5, 0],
                [0.5, 0, 0.5],
                [0, 0.5, 0.5],
            ]
        ),
    )

    assert fcc.get_coordination_number() == 12
    assert fcc.get_packing_efficiency() == 74
    assert fcc.get_radius() == 0.3535533905932738
