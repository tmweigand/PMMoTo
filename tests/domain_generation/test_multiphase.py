"""test_multiphase.py"""

import numpy as np
import pmmoto


def test_volume_fraction():
    """Ensures correct calculation of volume fraction"""
    sd = pmmoto.initialize(
        voxels=(10, 10, 10),
        boundary_types=(
            (pmmoto.BoundaryType.WALL, pmmoto.BoundaryType.WALL),
            (pmmoto.BoundaryType.WALL, pmmoto.BoundaryType.WALL),
            (pmmoto.BoundaryType.WALL, pmmoto.BoundaryType.WALL),
        ),
    )
    img = np.zeros(sd.voxels)
    img[3:6, 3:6, 3:6] = 1
    pm = pmmoto.domain_generation.porousmedia.gen_pm(sd, img)

    assert pm.porosity == (27 / 1000)

    mp_img = np.copy(img)
    mp_img[4:6, 4:6, 4:6] = 2
    mp = pmmoto.domain_generation.multiphase.Multiphase(
        porous_media=pm, img=mp_img, num_phases=2
    )

    assert mp.get_volume_fraction(1) == (19 / 1000)
    assert mp.get_volume_fraction(2) == (8 / 1000)

    assert mp.get_saturation(1) == (19 / 27)
    assert mp.get_saturation(2) == (8 / 27)


def test_get_radii():
    """Test if capillary pressures are converting to radii correctly"""
    p_c = 1
    gamma = 1
    radius = pmmoto.domain_generation.multiphase.Multiphase.get_probe_radius(p_c, gamma)
    assert radius == 2.0

    p_c = 2
    gamma = 1
    radius = pmmoto.domain_generation.multiphase.Multiphase.get_probe_radius(p_c, gamma)
    assert radius == 1.0

    p_c = 3
    gamma = 0.5
    radius = pmmoto.domain_generation.multiphase.Multiphase.get_probe_radius(p_c, gamma)
    assert radius == 1 / 3


def test_radius_with_contact_angle():
    """Test if capillary pressures are converting to radii with contact angle"""
    p_c = 1
    gamma = 1
    contact_angle = 20
    radius = pmmoto.domain_generation.multiphase.Multiphase.get_probe_radius(
        p_c, gamma, contact_angle
    )
    np.testing.assert_approx_equal(radius, 1.87938524)

    p_c = 2
    gamma = 1
    contact_angle = 20
    radius = pmmoto.domain_generation.multiphase.Multiphase.get_probe_radius(
        p_c, gamma, contact_angle
    )
    np.testing.assert_approx_equal(radius, 0.93969262)

    p_c = 3
    gamma = 0.5
    contact_angle = 85
    radius = pmmoto.domain_generation.multiphase.Multiphase.get_probe_radius(
        p_c, gamma, contact_angle
    )
    np.testing.assert_approx_equal(radius, 0.029051914)
