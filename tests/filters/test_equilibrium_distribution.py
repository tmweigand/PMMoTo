"""test_equilibrium_distribution.py"""

import pmmoto
import pytest
import numpy as np


def test_drainage_methods():
    """Ensures Drainage approaches similar with zero contact angle"""
    sphere_pack_file = "tests/test_data/sphere_packs/bcc.out"
    spheres, domain_box = pmmoto.io.data_read.read_sphere_pack_xyzr_domain(
        sphere_pack_file
    )

    voxels = (50, 50, 50)
    sd = pmmoto.initialize(voxels, box=domain_box)

    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres)
    mp = pmmoto.domain_generation.gen_mp_constant(pm, 2)

    capillary_pressure = 0.1

    std_method = pmmoto.filters.equilibrium_distribution.drainage(
        mp, capillary_pressure, contact_angle=0, method="standard"
    )

    ca_method = pmmoto.filters.equilibrium_distribution.drainage(
        mp, capillary_pressure, contact_angle=0, method="contact_angle"
    )

    eca_method = pmmoto.filters.equilibrium_distribution.drainage(
        mp, capillary_pressure, contact_angle=0, method="extended_contact_angle"
    )

    np.testing.assert_array_equal(std_method, ca_method)
    np.testing.assert_array_equal(std_method, eca_method)

    capillary_pressure = 20

    std_method = pmmoto.filters.equilibrium_distribution.drainage(
        mp, capillary_pressure, contact_angle=0, method="standard"
    )

    ca_method = pmmoto.filters.equilibrium_distribution.drainage(
        mp, capillary_pressure, contact_angle=0, method="contact_angle"
    )

    eca_method = pmmoto.filters.equilibrium_distribution.drainage(
        mp, capillary_pressure, contact_angle=0, method="extended_contact_angle"
    )

    np.testing.assert_array_equal(std_method, ca_method)
    np.testing.assert_array_equal(std_method, eca_method)


def test_drainage_save_img(tmpdir):
    """Tests saving of images"""
    sphere_pack_file = "tests/test_data/sphere_packs/bcc.out"
    spheres, domain_box = pmmoto.io.data_read.read_sphere_pack_xyzr_domain(
        sphere_pack_file
    )

    voxels = (50, 50, 50)
    sd = pmmoto.initialize(voxels, box=domain_box)

    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres)
    mp = pmmoto.domain_generation.gen_mp_constant(pm, 2)

    std_method = pmmoto.filters.equilibrium_distribution.drainage(
        mp, 0.1, contact_angle=0, method="standard", save=True, out_folder=tmpdir
    )


def test_drainage_errors():
    """Tests expected errors from bad input"""
    sphere_pack_file = "tests/test_data/sphere_packs/bcc.out"
    spheres, domain_box = pmmoto.io.data_read.read_sphere_pack_xyzr_domain(
        sphere_pack_file
    )

    voxels = (50, 50, 50)
    sd = pmmoto.initialize(voxels, box=domain_box)

    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres)
    mp = pmmoto.domain_generation.gen_mp_constant(pm, 2)

    with pytest.raises(ValueError):
        _ = pmmoto.filters.equilibrium_distribution.drainage(
            mp, 0.1, contact_angle=10, method="standard"
        )

    with pytest.raises(ValueError):
        _ = pmmoto.filters.equilibrium_distribution.drainage(
            mp, 0.1, contact_angle=0, method="undefined"
        )


def test_drainage_sorting(caplog):
    """Tests expected errors from bad input"""
    sphere_pack_file = "tests/test_data/sphere_packs/bcc.out"
    spheres, domain_box = pmmoto.io.data_read.read_sphere_pack_xyzr_domain(
        sphere_pack_file
    )

    voxels = (50, 50, 50)
    sd = pmmoto.initialize(voxels, box=domain_box)

    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres)
    mp = pmmoto.domain_generation.gen_mp_constant(pm, 2)

    cp = [2, 1]

    _ = pmmoto.filters.equilibrium_distribution.drainage(
        mp, cp, contact_angle=0, method="standard"
    )

    assert "must be monotonically increasing" in caplog.text
