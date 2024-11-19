import numpy as np
from mpi4py import MPI
import pmmoto
import time


def test_update_buffer():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

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
        rank=rank,
        mpi_size=size,
        reservoir_voxels=0,
    )

    grid = np.zeros(sd.voxels)
    own_nodes = [sd.voxels[0] - 2, sd.voxels[1] - 2, sd.voxels[2] - 2]
    grid[1:-1, 1:-1, 1:-1] = np.arange(
        own_nodes[0] * own_nodes[1] * own_nodes[2]
    ).reshape(own_nodes)

    updated_grid = pmmoto.core.communication.update_buffer(sd, grid)

    assert np.allclose(updated_grid, solution)

    MPI.Finalize()
