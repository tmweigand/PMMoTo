"""Unit tests for PMMoTo Statistics"""

import numpy as np
from mpi4py import MPI
import pytest
import pmmoto


def test_global_min() -> None:
    """Ensure global min is functional."""
    voxels = (100, 100, 100)
    sd = pmmoto.initialize(voxels=voxels)
    img = np.zeros(sd.voxels)

    img[0, 0, 0] = -3
    assert pmmoto.analysis.stats.get_minimum(sd, img) == -3
    assert pmmoto.analysis.stats.get_minimum(sd, img, own=False) == -3


def test_global_max() -> None:
    """Ensure global min is functional."""
    voxels = (100, 100, 100)
    sd = pmmoto.initialize(voxels=voxels)
    img = np.zeros(sd.voxels)

    img[0, 0, 0] = 3
    assert pmmoto.analysis.stats.get_maximum(sd, img) == 3
    assert pmmoto.analysis.stats.get_maximum(sd, img, own=False) == 3


@pytest.mark.mpi(min_size=8)
def test_global_min_parallel() -> None:
    """Ensure global min is functional."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    voxels = (100, 100, 100)
    subdomains = (2, 2, 2)

    sd = pmmoto.initialize(voxels=voxels, subdomains=subdomains, rank=rank)
    img = np.zeros(sd.voxels)

    img[3, 3, 3] = -rank
    assert pmmoto.analysis.stats.get_minimum(sd, img) == -7
    assert pmmoto.analysis.stats.get_minimum(sd, img, own=False) == -7


@pytest.mark.mpi(min_size=8)
def test_global_max_parallel() -> None:
    """Ensure global min is functional."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    voxels = (100, 100, 100)
    subdomains = (2, 2, 2)

    sd = pmmoto.initialize(voxels=voxels, subdomains=subdomains, rank=rank)
    img = np.zeros(sd.voxels)

    img[1, 1, 1] = rank
    assert pmmoto.analysis.stats.get_maximum(sd, img) == 7
    assert pmmoto.analysis.stats.get_minimum(sd, img, own=False) == 0
