"""Unit tests for connected components and inlet/outlet labeling in PMMoTo.

These tests cover parallel and serial connected components labeling,
periodic and non-periodic domains, and inlet-connected region detection.
"""

import numpy as np
from mpi4py import MPI
import pmmoto
import pytest


@pytest.mark.mpi(min_size=8)
def test_connect_components(
    generate_simple_subdomain: pmmoto.core.subdomain_padded.PaddedSubdomain,
) -> None:
    """Connected components labeling and inlet/outlet label detection in parallel."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    periodic = False
    sd = generate_simple_subdomain(0, periodic=periodic)

    img = np.ones(sd.domain.voxels)
    img[:, 5, :] = 0
    img[:, :, 5] = 0

    # if rank == 0:
    #     pmmoto.io.output.save_img(
    #         "data_out/cc_domain", img, resolution=sd.domain.resolution
    #     )

    subdomains = (2, 2, 2)
    sd_local, local_img = pmmoto.domain_generation.deconstruct_img(
        sd,
        img,
        subdomains=subdomains,
        rank=rank,
    )

    cc, label_count = pmmoto.filters.connected_components.connect_components(
        local_img, sd_local
    )

    # pmmoto.io.output.save_img(
    #     "data_out/test_cc", sd_local, local_img, additional_img={"cc": cc}
    # )

    connected_labels = pmmoto.filters.connected_components.inlet_outlet_labels(
        sd_local, cc
    )

    assert label_count == 5
    assert sorted(connected_labels) == [1, 2, 3, 4]


@pytest.mark.mpi(min_size=8)
def test_connect_components_periodic(
    generate_simple_subdomain: pmmoto.core.subdomain_padded.PaddedSubdomain,
) -> None:
    """Test connected components with periodic boundaries in parallel."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    periodic = True
    sd = generate_simple_subdomain(0, periodic=periodic)

    img = np.ones(sd.domain.voxels)
    img[:, 5, :] = 0
    img[:, :, 5] = 0

    subdomains = (2, 2, 2)
    sd_local, local_img = pmmoto.domain_generation.deconstruct_img(
        sd,
        img,
        subdomains=subdomains,
        rank=rank,
    )

    cc, label_count = pmmoto.filters.connected_components.connect_components(
        local_img, sd_local
    )

    connected_labels = pmmoto.filters.connected_components.inlet_outlet_connections(
        sd_local, cc
    )

    assert label_count == 2
    assert sorted(connected_labels) == []


def test_connect_components_bcs_0(
    generate_simple_subdomain: pmmoto.core.subdomain_padded.PaddedSubdomain,
) -> None:
    """Test connected components with background label 0. Should ignore background."""
    sd = generate_simple_subdomain(
        0,
        specified_types=(
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        ),
    )
    img = np.arange(np.prod(sd.domain.voxels)).reshape(sd.domain.voxels)

    subdomains = (1, 1, 1)
    sd_local, local_img = pmmoto.domain_generation.deconstruct_img(
        sd,
        img,
        subdomains=subdomains,
        rank=0,
    )

    cc, label_count = pmmoto.filters.connected_components.connect_components(
        local_img, sd_local
    )

    assert label_count == np.prod(sd.domain.voxels) - 1


def test_connect_components_bcs_1(
    generate_simple_subdomain: pmmoto.core.subdomain_padded.PaddedSubdomain,
) -> None:
    """Test connected components with all labels nonzero. Should count all voxels."""
    sd = generate_simple_subdomain(
        0,
        specified_types=(
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        ),
    )
    img = np.arange(np.prod(sd.domain.voxels)).reshape(sd.domain.voxels)
    img = img + 1

    subdomains = (1, 1, 1)
    sd_local, local_img = pmmoto.domain_generation.deconstruct_img(
        sd,
        img,
        subdomains=subdomains,
        rank=0,
    )

    cc, label_count = pmmoto.filters.connected_components.connect_components(
        local_img, sd_local
    )

    assert label_count == np.prod(sd.domain.voxels)


def test_connect_components_partial_periodic(
    generate_simple_subdomain: pmmoto.core.subdomain_padded.PaddedSubdomain,
) -> None:
    """Test connected components with partial periodic boundaries in serial."""
    p_x = (
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
    )
    p_y = (
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
    )
    p_z = (
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
    )

    p_xy = (
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
    )
    p_xz = (
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
    )

    for p in [p_x, p_y, p_z, p_xy, p_xz]:
        sd = generate_simple_subdomain(0, specified_types=p)
        img = np.arange(np.prod(sd.domain.voxels)).reshape(sd.domain.voxels)

        subdomains = (1, 1, 1)
        sd_local, local_img = pmmoto.domain_generation.deconstruct_img(
            sd,
            img,
            subdomains=subdomains,
            rank=0,
        )

        _, label_count = pmmoto.filters.connected_components.connect_components(
            local_img, sd_local
        )

        assert label_count == np.prod(sd.domain.voxels)


@pytest.mark.mpi(min_size=8)
def test_connect_components_partial_periodic_parallel(
    generate_simple_subdomain: pmmoto.core.subdomain_padded.PaddedSubdomain,
) -> None:
    """Connected components with partial and full periodic boundaries in parallel."""
    p_x = (
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
    )
    p_y = (
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
    )
    p_z = (
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
    )

    p_xy = (
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
    )
    p_xz = (
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
    )

    p_xyz = (
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    for p in [p_x, p_y, p_z, p_xy, p_xz, p_xyz]:
        sd = generate_simple_subdomain(0, specified_types=p)
        img = np.arange(np.prod(sd.domain.voxels)).reshape(sd.domain.voxels)

        subdomains = (2, 2, 2)
        sd_local, local_img = pmmoto.domain_generation.deconstruct_img(
            sd,
            img,
            subdomains=subdomains,
            rank=rank,
        )

        cc, max_label = pmmoto.filters.connected_components.connect_components(
            local_img, sd_local
        )

        assert max_label == np.prod(sd.domain.voxels)


def test_inlet_connected_img() -> None:
    """Test that only regions connected to the inlet are labeled as connected."""
    voxels = (20, 20, 20)
    inlet = ((1, 0), (0, 0), (0, 0))
    sd = pmmoto.initialize(voxels=voxels, inlet=inlet)

    img = np.zeros(sd.voxels, dtype=np.uint8)
    img[0:10, 5:10, 5:10] = 1
    img[12:15, 12:15, 12:15] = 1

    labeled_image = pmmoto.filters.connected_components.inlet_connected_img(sd, img)

    np.testing.assert_array_equal(labeled_image[12:15, 12:15, 12:15], 0)
    np.testing.assert_array_equal(labeled_image[0:10, 5:10, 5:10], 1)
