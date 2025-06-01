"""average.py

Averaging utilities for PMMoTo, including linear averages along a given direction.
"""

import numpy as np
from mpi4py import MPI
from pmmoto.core import utils

comm = MPI.COMM_WORLD

__all__ = ["linear"]


def linear(subdomain, data, direction):
    """Calculate the linear average along a given direction.

    Args:
        subdomain: Subdomain object.
        data (np.ndarray): Input data array.
        direction (int): Axis along which to compute the average.

    Returns:
        np.ndarray: Averaged values along the specified direction.

    """
    grid = utils.own_grid(data, subdomain.index_own_nodes)
    proc_map = subdomain.domain.global_map[1:-1, 1:-1, 1:-1]
    voxels = np.prod(subdomain.domain.nodes) / subdomain.domain.nodes[direction]

    _sum = grid
    not_direction = []
    for n in [0, 1, 2]:
        if n != direction:
            _sum = np.sum(_sum, n, keepdims=True)
            not_direction.append(n)
    vals = np.squeeze(_sum)

    if subdomain.size != 1:

        proc_vals = comm.gather(vals, root=0)
        ave = None

        if subdomain.ID == 0:
            ave = np.zeros(subdomain.domain.nodes[direction])
            index_start = subdomain.domain.sub_nodes[direction]
            i, j = np.nested_iters(
                proc_map, [[direction], not_direction], flags=["multi_index"]
            )
            for n, _ in enumerate(i):
                for proc in j:
                    if proc != -1:
                        start = int(n * index_start)
                        end = start + proc_vals[proc].shape[0]
                        ave[start:end] += proc_vals[proc]

        ave = comm.bcast(ave, root=0)

    else:
        ave = vals

    return ave / voxels
