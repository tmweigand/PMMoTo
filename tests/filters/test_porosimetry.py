"""test_porosimetry.py"""

import pytest
import pmmoto
import numpy as np


def test_porosimetry_sizes():
    """
    Test probe diamterr sizes for linear and logarithmic
    """
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
    print(values)
    # log test
    np.testing.assert_array_almost_equal(values, [10.0, 4.64158883, 2.15443469, 1.0])


@pytest.mark.xfail
def test_porosimetry_sizes_input_fail():
    """
    ensuring checks are behaving correctly
    """

    values = pmmoto.filters.porosimetry.get_sizes(5, 1, 5)
    values = pmmoto.filters.porosimetry.get_sizes(0, 10, 0)


def test_porosimetry(generate_simple_subdomain):
    """
    Single process test for porosimetry
    """

    radius = 0.03

    spheres, domain_data = pmmoto.io.data_read.read_sphere_pack_xyzr_domain(
        "tests/test_domains/bcc.in"
    )

    sd = generate_simple_subdomain(
        rank=0,
        box=domain_data,
        specified_types=((2, 2), (2, 2), (2, 2)),
        voxels_in=(100, 100, 100),
    )

    # generate porous media
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres)

    # calculate the euclidean distance transform
    edt = pmmoto.filters.distance.edt(img=pm.img, subdomain=sd)

    # morph addition
    morph_add = pmmoto.filters.morphological_operators.addition(
        subdomain=sd, img=pm.img, radius=radius, fft=False
    )

    # morph subtraction
    morph_subtract = pmmoto.filters.morphological_operators.subtraction(
        subdomain=sd, img=pm.img, radius=radius, fft=False
    )

    # Visualize the data
    pmmoto.io.output.save_img_data_parallel(
        "data_out/test_porosimetry",
        sd,
        pm.img,
        additional_img={"edt": edt, "add": morph_add, "subtract": morph_subtract},
    )
