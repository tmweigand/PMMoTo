import numpy as np
from mpi4py import MPI
import pmmoto
import cc3d
import pytest


@pytest.mark.mpi(min_size=8)
def test_connect_componets(generate_simple_subdomain):
    """ """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    periodic = False
    sd = generate_simple_subdomain(0, periodic=periodic)

    img = np.ones(sd.domain.voxels)
    img[:, 5, :] = 0
    img[:, :, 5] = 0

    subdomains = (2, 2, 2)
    sd_local, local_img = pmmoto.core.pmmoto.deconstruct_grid(
        sd,
        img,
        subdomains=subdomains,
        rank=rank,
    )

    cc, label_count = pmmoto.filters.connected_components.connect_components(
        local_img, sd_local, return_label_count=True
    )

    connected_labels = pmmoto.filters.connected_components.inlet_outlet_labels(
        sd_local, cc
    )

    assert label_count == 5
    assert sorted(connected_labels) == [1, 2, 3, 4]


@pytest.mark.mpi(min_size=8)
def test_connect_componets_periodic(generate_simple_subdomain):
    """ """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    periodic = True
    sd = generate_simple_subdomain(0, periodic=periodic)

    img = np.ones(sd.domain.voxels)
    img[:, 5, :] = 0
    img[:, :, 5] = 0

    subdomains = (2, 2, 2)
    sd_local, local_img = pmmoto.core.pmmoto.deconstruct_grid(
        sd,
        img,
        subdomains=subdomains,
        rank=rank,
    )

    cc, label_count = pmmoto.filters.connected_components.connect_components(
        local_img, sd_local, return_label_count=True
    )

    connected_labels = pmmoto.filters.connected_components.inlet_outlet_labels(
        sd_local, cc
    )

    assert label_count == 2
    assert sorted(connected_labels) == []


def test_connect_componets_bcs_0(generate_simple_subdomain):
    """
    This test calls pmmoto  but only depends on the cc3d dependency.
    The label count is 999 because 0 is background.
    """

    sd = generate_simple_subdomain(0, specified_types=((0, 0), (0, 0), (0, 0)))
    img = np.arange(np.prod(sd.domain.voxels)).reshape(sd.domain.voxels)

    subdomains = (1, 1, 1)
    sd_local, local_img = pmmoto.core.pmmoto.deconstruct_grid(
        sd,
        img,
        subdomains=subdomains,
        rank=0,
    )

    cc, label_count = pmmoto.filters.connected_components.connect_components(
        local_img, sd_local, return_label_count=True
    )

    assert label_count == np.prod(sd.domain.voxels) - 1


def test_connect_componets_bcs_1(generate_simple_subdomain):
    """
    Same as above but add 1 to img so label count is 1000
    """

    sd = generate_simple_subdomain(0, specified_types=((0, 0), (0, 0), (0, 0)))
    img = np.arange(np.prod(sd.domain.voxels)).reshape(sd.domain.voxels)
    img = img + 1

    subdomains = (1, 1, 1)
    sd_local, local_img = pmmoto.core.pmmoto.deconstruct_grid(
        sd,
        img,
        subdomains=subdomains,
        rank=0,
    )

    cc, label_count = pmmoto.filters.connected_components.connect_components(
        local_img, sd_local, return_label_count=True
    )

    assert label_count == np.prod(sd.domain.voxels)


def test_connect_componets_partial_periodic(generate_simple_subdomain):
    """ """
    p_x = ((2, 2), (0, 0), (0, 0))
    p_y = ((0, 0), (2, 2), (0, 0))
    p_z = ((0, 0), (0, 0), (2, 2))

    p_xy = ((2, 2), (2, 2), (0, 0))
    p_xz = ((2, 2), (0, 0), (2, 2))

    for p in [p_x, p_y, p_z, p_xy, p_xz]:
        sd = generate_simple_subdomain(0, specified_types=p)
        img = np.arange(np.prod(sd.domain.voxels)).reshape(sd.domain.voxels)
        img = img

        subdomains = (1, 1, 1)
        sd_local, local_img = pmmoto.core.pmmoto.deconstruct_grid(
            sd,
            img,
            subdomains=subdomains,
            rank=0,
        )

        cc, label_count = pmmoto.filters.connected_components.connect_components(
            local_img, sd_local, return_label_count=True
        )

        connected_labels = pmmoto.filters.connected_components.inlet_outlet_labels(
            sd_local, cc
        )

        assert label_count == np.prod(sd.domain.voxels)


@pytest.mark.mpi(min_size=8)
def test_connect_componets_partial_periodic(generate_simple_subdomain):
    """ """
    p_x = ((2, 2), (0, 0), (0, 0))
    p_y = ((0, 0), (2, 2), (0, 0))
    p_z = ((0, 0), (0, 0), (2, 2))

    p_xy = ((2, 2), (2, 2), (0, 0))
    p_xz = ((2, 2), (0, 0), (2, 2))

    p_xyz = ((2, 2), (2, 2), (2, 2))

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    for p in [p_x, p_y, p_z, p_xy, p_xz, p_xyz]:
        sd = generate_simple_subdomain(0, specified_types=p)
        img = np.arange(np.prod(sd.domain.voxels)).reshape(sd.domain.voxels)

        subdomains = (2, 2, 2)
        sd_local, local_img = pmmoto.core.pmmoto.deconstruct_grid(
            sd,
            img,
            subdomains=subdomains,
            rank=rank,
        )

        cc, max_label = pmmoto.filters.connected_components.connect_components(
            local_img, sd_local, return_label_count=True
        )

        assert max_label == np.prod(sd.domain.voxels)


def test_inlet_connected_img():
    """
    Test for passing in an image and only return where labels are on the inlet
    """
    voxels = (20, 20, 20)
    inlet = ((1, 0), (0, 0), (0, 0))
    sd = pmmoto.initialize(voxels=voxels, inlet=inlet)

    img = np.zeros(sd.voxels)
    img[0:10, 5:10, 5:10] = 1
    img[12:15, 12:15, 12:15] = 1

    labeled_image = pmmoto.filters.connected_components.inlet_connected_img(sd, img)

    np.testing.assert_array_equal(labeled_image[12:15, 12:15, 12:15], 0)
    np.testing.assert_array_equal(labeled_image[0:10, 5:10, 5:10], 1)
