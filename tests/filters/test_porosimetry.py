"""test_porosimetry.py"""

import pytest
import pmmoto
import numpy as np


def test_porosimetry_sizes():
    """Test probe diamterr sizes for linear and logarithmic"""
    min_value = 0
    max_value = 10
    num_values = 4
    values = pmmoto.filters.porosimetry.get_sizes(
        min_value, max_value, num_values, "linear"
    )

    # linear test
    np.testing.assert_array_almost_equal(values, [10.0, 6.666667, 3.333333, 0.0])

    min_log_value = 1
    values = pmmoto.filters.porosimetry.get_sizes(
        min_log_value, max_value, num_values, "log"
    )

    np.testing.assert_array_almost_equal(values, [10.0, 4.64158883, 2.15443469, 1.0])


def test_porosimetry_sizes_input_fail():
    """Ensuring checks are behaving correctly"""
    with pytest.raises(ValueError):
        _ = pmmoto.filters.porosimetry.get_sizes(5, 1, 5)

    with pytest.raises(ValueError):
        _ = pmmoto.filters.porosimetry.get_sizes(0, 10, 0)

    with pytest.raises(ValueError):
        _ = pmmoto.filters.porosimetry.get_sizes(0, 10, 5, spacing="log")

    with pytest.raises(ValueError):
        _ = pmmoto.filters.porosimetry.get_sizes(0, 10, 5, spacing="undefined")


def test_modes():
    """Test morph, dt, and hybrid porosimetry modes.

    Expected output is all should be equal to eachother.
    """
    radius = 0.030001
    spheres, domain_data = pmmoto.io.data_read.read_sphere_pack_xyzr_domain(
        "tests/test_domains/bcc.in"
    )
    sd = pmmoto.initialize(
        rank=0,
        box=domain_data,
        boundary_types=(
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        ),
        voxels=(100, 100, 100),
    )

    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres)
    morph = pmmoto.filters.porosimetry.porosimetry(sd, pm, radius, mode="morph")
    dt_mode = pmmoto.filters.porosimetry.porosimetry(sd, pm, radius, mode="distance")
    hybrid = pmmoto.filters.porosimetry.porosimetry(sd, pm, radius, mode="hybrid")
    diff_radii = pmmoto.filters.porosimetry.porosimetry(
        sd, pm, [radius, radius], mode="morph"
    )

    np.testing.assert_array_almost_equal(morph, dt_mode)
    np.testing.assert_array_almost_equal(morph, hybrid)
    np.testing.assert_array_almost_equal(morph, diff_radii)

    with pytest.raises(ValueError):
        _ = pmmoto.filters.porosimetry.porosimetry(
            sd, pm, [radius, radius, radius], mode="morph"
        )


def test_porosimetry_inlet():
    """Ensure when inlet = True, that only voxels connected in inlet are > 0"""
    radius = 0.02
    voxels = (100, 100, 100)
    inlet = ((1, 0), (0, 0), (0, 0))
    sd = pmmoto.initialize(voxels=voxels, inlet=inlet)

    img = np.zeros(sd.voxels)
    img[0:10, 5:10, 5:10] = 1
    img[12:35, 12:35, 12:35] = 1

    pm = pmmoto.domain_generation.porousmedia.gen_pm(subdomain=sd, img=img)

    morph_no_inlet = pmmoto.filters.porosimetry.porosimetry(
        subdomain=sd, porous_media=pm, radius=radius, mode="morph", inlet=False
    )

    morph_inlet = pmmoto.filters.porosimetry.porosimetry(
        subdomain=sd, porous_media=pm, radius=radius, mode="morph", inlet=True
    )

    assert not np.array_equal(morph_no_inlet, morph_inlet)


def test_porosimetry():
    """Test generation of pore size distribution"""
    voxels = (30, 30, 30)
    sd = pmmoto.initialize(voxels)
    spheres = np.array([[0.5, 0.5, 0.5, 0.1]])
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres, invert=True)

    radius = 0.03333333333333334

    hybrid_psd = pmmoto.filters.porosimetry.porosimetry(
        subdomain=sd, porous_media=pm, radius=radius, mode="hybrid"
    )
    dist_psd = pmmoto.filters.porosimetry.porosimetry(
        subdomain=sd, porous_media=pm, radius=radius, mode="distance"
    )
    morph_psd = pmmoto.filters.porosimetry.porosimetry(
        subdomain=sd, porous_media=pm, radius=radius, mode="morph"
    )

    np.testing.assert_array_equal(hybrid_psd, dist_psd)
    np.testing.assert_array_equal(hybrid_psd, morph_psd)


def test_pore_size_distribution():
    """Test generation of pore size distribution"""
    voxels = (30, 30, 30)
    sd = pmmoto.initialize(voxels)
    spheres = np.array([[0.5, 0.5, 0.5, 0.1]])
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres, invert=True)

    hybrid_psd = pmmoto.filters.porosimetry.pore_size_distribution(
        sd, pm, inlet=False, mode="hybrid"
    )
    dist_psd = pmmoto.filters.porosimetry.pore_size_distribution(
        sd, pm, inlet=False, mode="distance"
    )
    morph_psd = pmmoto.filters.porosimetry.pore_size_distribution(
        sd, pm, inlet=False, mode="morph"
    )

    np.testing.assert_array_equal(hybrid_psd, dist_psd)
    np.testing.assert_array_equal(hybrid_psd, morph_psd)

    radius = 0.03333333333333334
    float_dist_psd = pmmoto.filters.porosimetry.pore_size_distribution(
        sd, pm, radii=radius, inlet=False, mode="distance"
    )

    radius = [0.03333333333333334]
    list_dist_psd = pmmoto.filters.porosimetry.pore_size_distribution(
        sd, pm, radii=radius, inlet=False, mode="distance"
    )

    radius = np.array([0.03333333333333334])
    np_dist_psd = pmmoto.filters.porosimetry.pore_size_distribution(
        sd, pm, radii=radius, inlet=False, mode="distance"
    )

    np.testing.assert_array_equal(float_dist_psd, list_dist_psd)
    np.testing.assert_array_equal(float_dist_psd, np_dist_psd)


def test_plot_pore_size_distribution_pdf(tmp_path):
    """Test plot_pore_size_distribution with PDF mode"""
    voxels = (30, 30, 30)
    sd = pmmoto.initialize(voxels)
    spheres = np.array([[0.5, 0.5, 0.5, 0.1]])
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres, invert=True)

    pore_size_img = pmmoto.filters.porosimetry.pore_size_distribution(
        sd, pm, num_radii=2, inlet=False, mode="hybrid"
    )

    output_file = str(tmp_path / "test_output")
    pmmoto.filters.porosimetry.plot_pore_size_distribution(
        file_name=output_file,
        subdomain=sd,
        pore_size_image=pore_size_img,
        plot_type="pdf",
    )

    # Check that file was created (only on rank 0)
    if sd.rank == 0:
        expected_file = tmp_path / "test_output_pore_size_distribution.png"
        assert expected_file.exists()


def test_plot_pore_size_distribution_cdf(tmp_path):
    """Test plot_pore_size_distribution with CDF mode"""
    voxels = (30, 30, 30)
    sd = pmmoto.initialize(voxels)
    spheres = np.array([[0.5, 0.5, 0.5, 0.1]])
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres, invert=True)

    pore_size_img = pmmoto.filters.porosimetry.pore_size_distribution(
        sd, pm, num_radii=2, inlet=False, mode="hybrid"
    )

    output_file = str(tmp_path / "test_cdf_output")
    pmmoto.filters.porosimetry.plot_pore_size_distribution(
        file_name=output_file,
        subdomain=sd,
        pore_size_image=pore_size_img,
        plot_type="cdf",
    )

    # Check that file was created (only on rank 0)
    if sd.rank == 0:
        expected_file = tmp_path / "test_cdf_output_pore_size_distribution.png"
        assert expected_file.exists()
