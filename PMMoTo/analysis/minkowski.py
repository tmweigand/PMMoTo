"""minkowski.py"""
import numpy as np
from mpi4py import MPI
from pmmoto.analysis import _minkowski
from pmmoto.core import utils
comm = MPI.COMM_WORLD

__all__ = [
    "functionals"
    ]    


def functionals(subdomain,grid):
    """
    Calculate the minkowki functionals.

    Only want values for own_nodes, however, 
    algorithm skips last index for bcs. 
    So if not +1 external boundary, keep the padding
    """
    if subdomain.size == 1:
        functionals = _minkowski.functionals(grid.astype(bool),subdomain.domain.nodes,subdomain.domain.voxel)
        print(grid.size)
    else:
        _index_own_nodes = np.copy(subdomain.index_own_nodes)
        for face in [1,3,5]: # +1 external faces
            if subdomain.boundary_type[face] == -1: # Internal boundary
                _index_own_nodes[face] += 1

        _own_grid = utils.own_grid(grid,_index_own_nodes)
        functionals = _minkowski.functionals(_own_grid.astype(bool),subdomain.domain.nodes,subdomain.domain.voxel)

    return functionals