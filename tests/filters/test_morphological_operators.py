"""test_morphology.py"""

import scipy.ndimage
import numpy as np
import pmmoto
from mpi4py import MPI
import pytest


def test_gen_struct_ratio():
    """
    Convert sphere radius to num of voxels.
    """

    struct_ratio = pmmoto.filters.morphological_operators.gen_struct_ratio(
        resolution=[0.5, 0.02, 1.5], radius=1
    )

    np.testing.assert_equal(struct_ratio, [2, 50, 1])

    struct_ratio = pmmoto.filters.morphological_operators.gen_struct_ratio(
        [0.5, 0.02, 1.5], 10
    )

    np.testing.assert_equal(struct_ratio, [20, 500, 7])


def test_gen_struct_element():
    """
    Generate a spherical/circular structuring element
    """

    _, struct_element = pmmoto.filters.morphological_operators.gen_struct_element(
        resolution=[0.5, 0.02, 0.25], radius=1
    )

    # struct_element shape is equal to 2*struct_ratio + 1
    assert struct_element.shape == (5, 101, 9)
    assert np.sum(struct_element) == 1569

    _, struct_element = pmmoto.filters.morphological_operators.gen_struct_element(
        resolution=[0.05, 0.02, 0.25], radius=0.1
    )

    # struct_element shape is equal to 2*struct_ratio + 1
    assert struct_element.shape == (5, 11, 3)
    assert np.sum(struct_element) == 31


@pytest.mark.mpi(min_size=8)
def test_morphological_addition(generate_simple_subdomain):
    """
    Generate a spherical/circular structuring element
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    probe_radius = 0.05
    subdomains = (2, 2, 2)

    spheres, domain_data = pmmoto.io.data_read.read_sphere_pack_xyzr_domain(
        "tests/test_domains/bcc.in"
    )

    for boundary_type in [0, 1, 2]:
        boundary = ((boundary_type, boundary_type),) * 3
        sd = generate_simple_subdomain(
            rank=0,
            specified_types=boundary,
            voxels_in=(100, 100, 100),
        )

        pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres)

        morphological_operator(
            "addition",
            rank,
            sd,
            pm.img,
            subdomains,
            probe_radius,
            boundary_type,
            test_scipy=True,
        )


@pytest.mark.mpi(min_size=8)
def test_morphological_subtraction(generate_simple_subdomain):
    """
    Generate a spherical/circular structuring element
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    subdomains = (2, 2, 2)
    probe_radius = 0.05

    spheres, domain_data = pmmoto.io.data_read.read_sphere_pack_xyzr_domain(
        "tests/test_domains/bcc.in"
    )

    for boundary_type in [0, 1, 2]:
        boundary = ((boundary_type, boundary_type),) * 3
        sd = generate_simple_subdomain(
            rank=0,
            box=domain_data,
            specified_types=boundary,
            voxels_in=(100, 100, 100),
        )

        pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres)

        morphological_operator(
            "subtraction",
            rank,
            sd,
            pm.img,
            subdomains,
            probe_radius,
            boundary_type,
            test_scipy=False,
        )


def morphological_operator(
    operator, rank, sd, img, subdomains, probe_radius, boundary_type, test_scipy=False
):
    """
    Wrapper for more efficient tests
    """

    if operator == "addition":
        morph_operator = pmmoto.filters.morphological_operators.addition
        scipy_operator = scipy.ndimage.binary_dilation
    elif operator == "subtraction":
        morph_operator = pmmoto.filters.morphological_operators.subtraction
        scipy_operator = scipy.ndimage.binary_erosion
    else:
        raise ValueError(f"Operator mus be 'addition' or 'subtraction' not {operator}")

    # pmmoto result
    morph_result = morph_operator(subdomain=sd, img=img, radius=probe_radius, fft=False)

    # remove pad
    morph_result = pmmoto.core.utils.unpad(morph_result, sd.pad)
    img_base = pmmoto.core.utils.unpad(img, sd.pad)

    if test_scipy:
        _, struct_element = pmmoto.filters.morphological_operators.gen_struct_element(
            resolution=sd.domain.resolution, radius=probe_radius
        )

        scipy_morph_result = scipy_operator(img_base, struct_element)

        np.testing.assert_array_almost_equal(
            morph_result,
            scipy_morph_result,
        )

    sd_local, local_img = pmmoto.core.pmmoto.deconstruct_grid(
        sd,
        img_base,
        subdomains=subdomains,
        rank=rank,
    )

    sd_local, local_morph_result = pmmoto.core.pmmoto.deconstruct_grid(
        sd,
        morph_result,
        subdomains=subdomains,
        rank=rank,
    )

    morph_img_edt = morph_operator(
        subdomain=sd_local, img=local_img, radius=probe_radius, fft=False
    )

    morph_img_fft = morph_operator(
        subdomain=sd_local, img=local_img, radius=probe_radius, fft=True
    )

    np.testing.assert_array_almost_equal(morph_img_fft, local_morph_result)
    np.testing.assert_array_almost_equal(morph_img_edt, local_morph_result)
    np.testing.assert_array_almost_equal(morph_img_edt, morph_img_fft)
