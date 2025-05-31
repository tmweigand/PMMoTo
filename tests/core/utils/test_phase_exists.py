"""test_phase_exists.py"""

from mpi4py import MPI
import numpy as np
import pmmoto

import pytest


@pytest.mark.skip(reason="TBD")
def test_phase_exists():
    """Check to make sure utils.phase_exists works properly
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    grid = np.zeros([5, 5, 5], dtype=np.uint8)

    assert pmmoto.core.utils.phase_exists(grid, 0)
    assert not pmmoto.core.utils.phase_exists(grid, 1)

    if rank == 0:
        grid = 2 * np.ones_like(grid)

    assert pmmoto.core.utils.phase_exists(grid, 2)
