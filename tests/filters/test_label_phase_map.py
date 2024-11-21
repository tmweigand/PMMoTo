import numpy as np
from mpi4py import MPI
import pmmoto
import time

import pytest


@pytest.mark.skip(reason="TBD")
def test_label_phase_map():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomain_map = [1, 1, 1]
    voxels = [3, 3, 3]

    box = [[0, 1], [0, 1], [0, 1]]
    boundaries = [[2, 2], [2, 2], [2, 2]]
    inlet = [[0, 0], [0, 0], [0, 0]]
    outlet = [[0, 0], [0, 0], [0, 0]]

    sd, domain = pmmoto.initialize(
        box=box,
        subdomain_map=subdomain_map,
        voxels=voxels,
        boundaries=boundaries,
        inlet=inlet,
        outlet=outlet,
        rank=rank,
        mpi_size=size,
        reservoir_voxels=0,
    )

    grid = np.zeros(sd.voxels, dtype=np.uint64)
    own_nodes = [sd.voxels[0] - 2, sd.voxels[1] - 2, sd.voxels[2] - 2]
    grid[1:-1, 1:-1, 1:-1] = np.arange(
        own_nodes[0] * own_nodes[1] * own_nodes[2]
    ).reshape(own_nodes)
    updated_grid = pmmoto.core.communication.update_buffer(sd, grid)
    updated_grid_8 = updated_grid.astype(np.uint8)
    phase_map = pmmoto.filters.get_label_phase_map(updated_grid_8, updated_grid)
    print(phase_map)

    phase_count = pmmoto.filters.phase_count(phase_map)
    print(phase_count)


if __name__ == "__main__":
    test_label_phase_map()
    MPI.Finalize()
