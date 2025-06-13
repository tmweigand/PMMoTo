"""Unit tests for PMMoTo core communication routines.

Tests include buffer updates, feature communication, and buffer extension
with MPI parallelism.
"""

import numpy as np
from mpi4py import MPI
import pytest
import pmmoto


def test_update_buffer() -> None:
    """Ensure that features and buffer are being communicated to neighbor processes"""
    solution = np.array(
        [
            [
                [26, 24, 25, 26, 24],
                [20, 18, 19, 20, 18],
                [23, 21, 22, 23, 21],
                [26, 24, 25, 26, 24],
                [20, 18, 19, 20, 18],
            ],
            [
                [8, 6, 7, 8, 6],
                [2, 0, 1, 2, 0],
                [5, 3, 4, 5, 3],
                [8, 6, 7, 8, 6],
                [2, 0, 1, 2, 0],
            ],
            [
                [17, 15, 16, 17, 15],
                [11, 9, 10, 11, 9],
                [14, 12, 13, 14, 12],
                [17, 15, 16, 17, 15],
                [11, 9, 10, 11, 9],
            ],
            [
                [26, 24, 25, 26, 24],
                [20, 18, 19, 20, 18],
                [23, 21, 22, 23, 21],
                [26, 24, 25, 26, 24],
                [20, 18, 19, 20, 18],
            ],
            [
                [8, 6, 7, 8, 6],
                [2, 0, 1, 2, 0],
                [5, 3, 4, 5, 3],
                [8, 6, 7, 8, 6],
                [2, 0, 1, 2, 0],
            ],
        ],
        dtype=int,
    )

    subdomains = (1, 1, 1)
    voxels = (3, 3, 3)
    box = ((0, 1), (0, 1), (0, 1))
    boundary_types = (
        ("periodic", "periodic"),
        ("periodic", "periodic"),
        ("periodic", "periodic"),
    )

    sd = pmmoto.initialize(
        box=box,
        subdomains=subdomains,
        voxels=voxels,
        boundary_types=boundary_types,
        rank=0,
    )

    img = np.zeros(sd.voxels)
    own_nodes = [sd.voxels[0] - 2, sd.voxels[1] - 2, sd.voxels[2] - 2]
    img[1:-1, 1:-1, 1:-1] = np.arange(
        own_nodes[0] * own_nodes[1] * own_nodes[2]
    ).reshape(own_nodes)

    updated_grid = pmmoto.core.communication.update_buffer(sd, img)

    np.testing.assert_array_almost_equal(updated_grid, solution)


@pytest.mark.mpi(min_size=8)
def test_communicate_features() -> None:
    """Ensure that features are being communicated to neighbor processes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    sd = pmmoto.initialize(
        box=((0, 1), (0, 1), (0, 1)),
        subdomains=(2, 2, 2),
        voxels=(10, 10, 10),
        boundary_types=(
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        ),
        rank=rank,
    )

    feature_data = {}
    for feature_id, feature in sd.features.all_features:
        feature_data[feature_id] = rank

    recv_data = pmmoto.core.communication.communicate_features(
        subdomain=sd,
        send_data=feature_data,
    )

    for feature_id, feature in sd.features.all_features:
        if feature_id in recv_data.keys():
            assert recv_data[feature_id] == feature.neighbor_rank


@pytest.mark.mpi(min_size=8)
def test_update_buffer_with_buffer() -> None:
    """Ensure that features are being communicated to neighbor processes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    sd = pmmoto.initialize(
        box=((0, 1), (0, 1), (0, 1)),
        subdomains=(2, 2, 2),
        voxels=(10, 10, 10),
        boundary_types=(
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        ),
        rank=rank,
        pad=(1, 1, 1),
    )

    img = (rank + 1) * np.ones(sd.voxels)
    img = sd.set_wall_bcs(img)

    buffer = (2, 2, 2)

    update_img, halo = pmmoto.core.communication.update_extended_buffer(
        subdomain=sd,
        img=img,
        buffer=buffer,
    )

    pmmoto.io.output.save_img_data_parallel(
        "data_out/test_comm_buffer", sd, img, additional_img={"og": img}
    )

    pmmoto.io.output.save_extended_img_data_parallel(
        "data_out/test_comm_buffer_extended", sd, update_img, halo
    )


@pytest.mark.mpi(min_size=8)
def test_update_buffer_with_buffer() -> None:
    """Ensure that features are being communicated to neighbor processes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    box = (
        (0.0, 176),
        (0.0, 176),
        (-100, 100),
    )

    sd = pmmoto.initialize(
        box=box,
        subdomains=(2, 2, 2),
        voxels=(100, 100, 100),
        boundary_types=(
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        ),
        rank=rank,
        pad=(1, 1, 1),
    )

    img = (rank + 1) * np.ones(sd.voxels)
    img = sd.set_wall_bcs(img)

    buffer = (10, 10, 9)

    update_img, halo = pmmoto.core.communication.update_extended_buffer(
        subdomain=sd,
        img=img,
        buffer=buffer,
    )

    print(rank, halo, np.max(update_img))

    pmmoto.io.output.save_img_data_parallel(
        "data_out/test_comm_buffer", sd, img, additional_img={"og": img}
    )

    pmmoto.io.output.save_extended_img_data_parallel(
        "data_out/test_comm_buffer_extended", sd, update_img, halo
    )
