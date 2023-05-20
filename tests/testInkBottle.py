import numpy as np
from mpi4py import MPI
import pdb
import sys
import time
import PMMoTo
import math
import edt
from scipy.ndimage import distance_transform_edt
from line_profiler import LineProfiler


def my_function():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank==0:
        start_time = time.time()

    subDomains = [2,2,2]
    nodes = [280,60,60]
    #nodes = [560,120,120]
    #nodes = [840,180,180]
    #nodes = [1120,240,240]
    #nodes = [2240,481,481]
    periodic = [False,False,False]
    inlet  = [ 1,0,0]
    outlet = [-1,0,0]

    #pC = [7.69409]
    pC = [7.69409,7.69409,7.69393,7.60403,7.36005,6.9909,6.53538,6.03352,5.52008,5.02123,
          4.55421,4.12854,3.74806,3.41274,3.12024,2.86704,2.64914,2.4625,2.30332,2.16814,2.05388,
          1.95783,1.87764,1.81122,1.75678,1.7127,1.67755,1.65002,1.62893,1.61322,1.60194,
          1.5943,1.58965,1.58964,1.5872,0.]

    numSubDomains = np.prod(subDomains)

    drain = True
    test = False

    domain,sDL = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,periodic,inlet,outlet,"InkBotle",None,None)
    sDEDTL = PMMoTo.calcEDT(rank,size,domain,sDL,sDL.grid)
    if drain:
        drainL = PMMoTo.calcDrainage(rank,size,pC,domain,sDL,inlet,sDEDTL)


# pC = [7.69409,7.69409,7.69393,7.60403,7.36005,6.9909,6.53538,6.03352,5.52008,5.02123,
# 4.55421,4.12854,3.74806,3.41274,3.12024,2.86704,2.64914,2.4625,2.30332,2.16814,2.05388,
# 1.95783,1.87764,1.81122,1.75678,1.7127,1.67755,1.65002,1.62893,1.61322,1.60194,
# 1.5943,1.58965,1.58964,1.5872,0.]
#
# sWAnalytical = [0.,0.808933,0.808965,0.809662,0.810321,0.810973,0.811641,0.81235,0.813122,
#                 0.813978,0.814941,0.816034,0.817281,0.818703,0.820323,0.822162,0.824238,
#                 0.826565,0.829154,0.832009,0.835131,0.838511,0.842134,0.845978,0.850016,
#                 0.854214,0.858539,0.862957,0.867443,0.871985,0.876591,0.881302,0.886197,1.,1.,1.]
#




if __name__ == "__main__":
    my_function()
