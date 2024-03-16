"""stats.py"""
import numpy as np
from mpi4py import MPI
from pmmoto.core import utils

comm = MPI.COMM_WORLD

__all__ = [
    "get_minimum",
    "get_maximum",
    "get_volume_fraction",
    "get_saturation"
]

def get_minimum(subdomain,data):
    """
    Determine the global minimum of the data
    """
    if subdomain.size > 1:
        own_data = utils.own_grid(data,subdomain.index_own_nodes)
        local_min = np.min(own_data)
        global_min = comm.allreduce(local_min,op = MPI.MIN)
        return global_min
    else:
        return np.min(data)

def get_maximum(subdomain,data):
    """
    Determine the global maximum of the data
    """
    if subdomain.size > 1:
        own_data = utils.own_grid(data,subdomain.index_own_nodes)
        local_max = np.max(own_data)
        global_max = comm.allreduce(local_max,op = MPI.MAX)
        return global_max
    else:
        return np.max(data)

def get_volume_fraction(subdomain,data,phase):
    """
    Calculate the volume fraction of a given phase
    """
    own_grid = utils.own_grid(data,subdomain.index_own_nodes)
    local_phase_nodes = np.count_nonzero(own_grid == phase)
    global_phase_nodes = comm.allreduce(local_phase_nodes,op = MPI.SUM )
    volume_fraction = global_phase_nodes/np.prod(subdomain.domain.nodes)
    return volume_fraction

def get_saturation(subdomain,data,phase,porousmedia = None):
    """
    Calculate the saturation of a given phase
    """

    volume_fraction = get_volume_fraction(subdomain,data,phase)

    if porousmedia is not None:
        if porousmedia.porosity is None:
            porousmedia.get_porosity()
        porosity = porousmedia.porosity
    else:
        porosity = 1. - get_volume_fraction(subdomain,data,0)
    
    return volume_fraction/porosity



# def genStats(subdomain,data):
#     """
#     Get Information (non-zero min/max) of distance tranform
#     """
#     own_data = utils.own_grid(data,subdomain.index_own_nodes)
#     vals,counts  = np.unique(own_data,return_counts=True)

#     EDTData = [subdomain.ID,vals,counts]
#     EDTData = comm.gather(EDTData, root=0)
#     if subdomain.ID == 0:
#         bins = np.empty([])
#         for d in EDTData:
#             if d[0] == 0:
#                 bins = d[1]
#             else:
#                 bins = np.append(bins,d[1],axis=0)
#             bins = np.unique(bins)

#         counts = np.zeros_like(bins,dtype=np.int64)
#         for d in EDTData:
#             for n in range(0,d[1].size):
#                 ind = np.where(bins==d[1][n])[0][0]
#                 counts[ind] = counts[ind] + d[2][n]

#         stats = np.stack((bins,counts), axis = 1)
#         data_min = bins[1]
#         data_max = bins[-1]
#         distData = [data_min,data_max]
#         print("Minimum distance:",data_min,"Maximum distance:",data_max)
#     else:
#         distData = None
#     distData = comm.bcast(distData, root=0)
#     data_min = distData[0]
#     data_max = distData[1]



# def generate_histogram():