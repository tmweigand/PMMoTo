import numpy as np
from mpi4py import MPI
import time
import PMMoTo


def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subDomains = [2,2,2]
    nodes = [151,151,151]

    ## Ordering for Inlet/Outlet ( (+x,-x) , (+y,-y) , (+z,-z) )
    boundaries = [[0,0],[0,0],[0,0]]
    inlet  = [[0,0],[0,0],[0,0]]
    outlet = [[0,0],[0,0],[0,0]]


    file = './testDomains/50pack.out'

    pC = [140,160]

    startTime = time.time()
    domain,sDL = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,"Sphere",file,PMMoTo.readPorousMediaXYZR)

    numFluidPhases = 2

    wRes  = [[0,1],[0,0],[0,0]]
    nwRes = [[1,0],[0,0],[0,0]]
    mpInlets = [wRes,nwRes]

    wOut  = [[0,0],[0,0],[0,0]]
    nwOut = [[0,0],[0,0],[0,0]]
    mpOutlets = [wOut,nwOut]

    twoPhase = PMMoTo.multiPhase.multiPhase(domain,sDL,numFluidPhases)
    twoPhase.initializeMPGrid(constantPhase = 1)
    twoPhase.getBoundaryInfo(mpInlets,mpOutlets)

    drainL = PMMoTo.multiPhase.calcDrainage(pC,twoPhase)

    endTime = time.time()
    print("Parallel Time:",endTime-startTime)


if __name__ == "__main__":
    my_function()
    MPI.Finalize()
