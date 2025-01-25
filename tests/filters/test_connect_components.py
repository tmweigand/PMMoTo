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
    assert connected_labels == [0, 1, 2, 3, 4]


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
    assert connected_labels == [0, 1]
