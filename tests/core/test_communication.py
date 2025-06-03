"""Unit tests for PMMoTo core communication routines.

Tests include buffer updates, feature communication, and buffer extension
with MPI parallelism.
"""

import numpy as np
from mpi4py import MPI
import pytest
import pmmoto


def test_update_buffer():
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
    voxels = [3, 3, 3]
    box = [[0, 1], [0, 1], [0, 1]]
    boundary_types = [[2, 2], [2, 2], [2, 2]]
    inlet = [[0, 0], [0, 0], [0, 0]]
    outlet = [[0, 0], [0, 0], [0, 0]]

    sd = pmmoto.initialize(
        box=box,
        subdomains=subdomains,
        voxels=voxels,
        boundary_types=boundary_types,
        inlet=inlet,
        outlet=outlet,
        rank=0,
        reservoir_voxels=0,
    )

    img = np.zeros(sd.voxels)
    own_nodes = [sd.voxels[0] - 2, sd.voxels[1] - 2, sd.voxels[2] - 2]
    img[1:-1, 1:-1, 1:-1] = np.arange(
        own_nodes[0] * own_nodes[1] * own_nodes[2]
    ).reshape(own_nodes)

    updated_grid = pmmoto.core.communication.update_buffer(sd, img)

    np.testing.assert_array_almost_equal(updated_grid, solution)


@pytest.mark.mpi(min_size=8)
def test_communicate_features():
    """Ensure that features are being communicated to neighbor processes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    sd = pmmoto.initialize(
        box=((0, 1), (0, 1), (0, 1)),
        subdomains=(2, 2, 2),
        voxels=(10, 10, 10),
        boundary_types=((2, 2), (2, 2), (2, 2)),
        rank=rank,
    )

    feature_data = {}
    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in sd.features[feature_type].items():
            feature_data[feature_id] = rank

    recv_data = pmmoto.core.communication.communicate_features(
        subdomain=sd,
        send_data=feature_data,
        feature_types=feature_types,
        unpack=True,
    )

    for feature_type in feature_types:
        for feature_id, feature in sd.features[feature_type].items():
            if feature_id in recv_data.keys():
                assert recv_data[feature_id] == feature.neighbor_rank


@pytest.mark.mpi(min_size=8)
def test_update_buffer_with_buffer():
    """Ensure that features are being communicated to neighbor processes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    sd = pmmoto.initialize(
        box=((0, 1), (0, 1), (0, 1)),
        subdomains=(2, 2, 2),
        voxels=(10, 10, 10),
        # boundary_types=((2, 2), (2, 2), (2, 2)),
        # boundary_types=((1, 1), (1, 1), (1, 1)),
        boundary_types=((0, 0), (0, 0), (0, 0)),
        rank=rank,
        pad=(1, 1, 1),
    )

    img = (rank + 1) * np.ones(sd.voxels)
    img = sd.set_wall_bcs(img)

    buffer = (2, 2, 2)

    update_img, halo = pmmoto.core.communication.update_buffer(
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
