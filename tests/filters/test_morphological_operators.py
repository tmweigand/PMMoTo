"""test_morphology.py"""

import scipy.ndimage
import numpy as np
import pmmoto
from mpi4py import MPI
import pytest


def test_gen_struct_ratio() -> None:
    """Convert sphere radius to num of voxels."""
    struct_ratio = pmmoto.filters.morphological_operators.gen_struct_ratio(
        resolution=[0.5, 0.5, 0.5], radius=1
    )

    np.testing.assert_equal(struct_ratio, [2, 2, 2])


def test_gen_struct_element() -> None:
    """Generate a spherical/circular structuring element"""
    _, struct_element = pmmoto.filters.morphological_operators.gen_struct_element(
        resolution=[0.01, 0.01, 0.01], radius=0.03
    )

    assert np.sum(struct_element) == 123


@pytest.mark.mpi(min_size=8)
def test_morphological_addition() -> None:
    """Generate a spherical/circular structuring element"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    probe_radius = 0.05
    subdomains = (2, 2, 2)

    spheres, domain_data = pmmoto.io.data_read.read_sphere_pack_xyzr_domain(
        "tests/test_domains/bcc.in"
    )

    for boundary_type in [
        pmmoto.BoundaryType.END,
        pmmoto.BoundaryType.WALL,
        pmmoto.BoundaryType.PERIODIC,
    ]:
        boundary = ((boundary_type, boundary_type),) * 3
        sd = pmmoto.initialize(
            rank=0,
            boundary_types=boundary,
            voxels=(100, 100, 100),
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
def test_morphological_subtraction() -> None:
    """Generate a spherical/circular structuring element"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    subdomains = (2, 2, 2)
    probe_radius = 0.05

    spheres, domain_data = pmmoto.io.data_read.read_sphere_pack_xyzr_domain(
        "tests/test_domains/bcc.in"
    )

    for boundary_type in [
        pmmoto.BoundaryType.END,
        pmmoto.BoundaryType.WALL,
        pmmoto.BoundaryType.PERIODIC,
    ]:
        boundary = ((boundary_type, boundary_type),) * 3
        sd = pmmoto.initialize(
            rank=0,
            box=domain_data,
            boundary_types=boundary,
            voxels=(100, 100, 100),
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


def test_morph_methods():
    """Test to Correctness of Different Approaches"""
    voxels = (30, 30, 30)
    sd = pmmoto.initialize(voxels)
    spheres = np.array([[0.5, 0.5, 0.5, 0.1]])
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres, invert=True)

    radius = 0.03333333333333334

    morph_fft = pmmoto.filters.morphological_operators.subtraction(
        subdomain=sd, img=pm.img, radius=radius, fft=True
    )

    morph_no_fft = pmmoto.filters.morphological_operators.subtraction(
        subdomain=sd, img=pm.img, radius=radius, fft=False
    )

    np.testing.assert_array_equal(morph_fft, morph_no_fft)

    morph_fft = pmmoto.filters.morphological_operators.addition(
        subdomain=sd, img=pm.img, radius=radius, fft=True
    )

    morph_no_fft = pmmoto.filters.morphological_operators.addition(
        subdomain=sd, img=pm.img, radius=radius, fft=False
    )

    np.testing.assert_array_equal(morph_fft, morph_no_fft)


def morphological_operator(
    operator, rank, sd, img, subdomains, probe_radius, boundary_type, test_scipy=False
) -> None:
    """Morphological test helper"""
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

    sd_local, local_img = pmmoto.domain_generation.deconstruct_img(
        sd,
        img_base,
        subdomains=subdomains,
        rank=rank,
    )

    sd_local, local_morph_result = pmmoto.domain_generation.deconstruct_img(
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
