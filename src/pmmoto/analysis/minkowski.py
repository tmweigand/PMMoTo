"""minkowski.py

Minkowski functionals analysis for PMMoTo.
"""

import numpy as np
from mpi4py import MPI
from pmmoto.analysis import _minkowski
from pmmoto.core import utils

comm = MPI.COMM_WORLD

__all__ = ["functionals"]


def functionals(subdomain, grid):
    """Calculate the Minkowski functionals for a subdomain.

    Only values for own_nodes are returned.
    The algorithm skips the last index for boundary conditions.
    If not a +1 external boundary, keep the padding.

    Args:
        subdomain: Subdomain object.
        grid (np.ndarray): Input binary grid.

    Returns:
        np.ndarray: Array of Minkowski functionals.

    """
    if subdomain.size == 1 and all(subdomain.boundary_type != 2):
        _functionals = _minkowski.functionals(
            grid.astype(bool), subdomain.domain.nodes, subdomain.domain.voxel
        )
    else:
        _index_own_nodes = np.copy(subdomain.index_own_nodes)
        for face in [1, 3, 5]:  # +1 external faces
            if subdomain.boundary_type[face] == -1:  # Internal boundary
                _index_own_nodes[face] += 1

        _own_grid = utils.own_grid(grid, _index_own_nodes)
        _functionals = _minkowski.functionals(
            _own_grid.astype(bool), subdomain.domain.nodes, subdomain.domain.voxel
        )

    return _functionals
