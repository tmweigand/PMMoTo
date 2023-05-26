import numpy as np
from mpi4py import MPI
import time
import PMMoTo



def my_function():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank==0:
        start_time = time.time()

    subDomains = [2,2,2]
    #nodes = [280,60,60]
    nodes = [560,120,120]
    #nodes = [840,180,180]
    #nodes = [1120,240,240]
    #nodes = [2240,481,481]

    boundaries = [[0,0],[1,1],[1,1]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[0,0]]
    outlet = [[0,0],[0,0],[0,0]]

    domain,sDL = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,"InkBotle",None,None)

    numFluidPhases = 2
    twoPhase = PMMoTo.multiPhase.multiPhase(domain,sDL,numFluidPhases)

    wRes  = [[1,0],[0,0],[0,0]]
    nwRes = [[0,1],[0,0],[0,0]]
    mpInlets = {twoPhase.wID:wRes,twoPhase.nwID:nwRes}

    wOut  = [[0,0],[0,0],[0,0]]
    nwOut = [[0,0],[0,0],[0,0]]
    mpOutlets = {twoPhase.wID:wOut,twoPhase.nwID:nwOut}
    
    #Initialize wetting saturated somain
    twoPhase.initializeMPGrid(constantPhase = twoPhase.wID) 
    twoPhase.getBoundaryInfo(mpInlets,mpOutlets)
    
    pC = [1.67755]
    drainL = PMMoTo.multiPhase.calcDrainage(pC,twoPhase)







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
