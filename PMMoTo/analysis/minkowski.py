"""minkowski.py"""

import numpy as np
from pmmoto.analysis import _minkowski


__all__ = [
    "test"
    ]    


def test(subdomain,grid):
    """
    
    """
    t = _minkowski.functionals(grid.astype(bool),subdomain.domain.voxel)
    return t


# def minkowskiEval(EDT,res=None):      
#     # evaluate grid minus last buffer entry to prevent double counting
#     # this may result in bad behavior for boundaries = 0...  
#     if (res[0] != res[1]) or (res[0] != res[2]):
#         print('In order to perform MF evaluation, isotropy is expected.')
#         return [None]
#     dist,volume,surface,curvature,euler = mk.functions_open(EDT[:-1,:-1,:-1]/res[0])
#     return [dist,volume,surface,curvature,euler]