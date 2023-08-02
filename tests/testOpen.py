import numpy as np
from mpi4py import MPI
import time
import PMMoTo


def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subDomains = [2,2,2] # Specifies how Domain is broken among rrocs
    nodes = [150,150,150] # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[2,2],[2,2],[2,2]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[0,0]]
    outlet = [[0,0],[0,0],[0,0]]

    ####CHANGE TO TARGET FILE
    file = './testDomains/50pack.out'


    startTime = time.time()
    domain,sDL,pML = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,"Sphere",file,PMMoTo.readPorousMediaXYZR)

    numFluidPhases = 2
    twoPhase = PMMoTo.multiPhase.multiPhase(pML,numFluidPhases)

    wRes  = [[0,0],[0,0],[0,0]]
    nwRes = [[0,0],[0,0],[0,0]]
    mpInlets = {twoPhase.wID:wRes,twoPhase.nwID:nwRes}

    wOut  = [[0,0],[0,0],[0,0]]
    nwOut = [[0,0],[0,0],[0,0]]
    mpOutlets = {twoPhase.wID:wOut,twoPhase.nwID:nwOut}

    #Initialize wetting saturated somain
    twoPhase.initializeMPGrid(constantPhase = twoPhase.wID) 

    twoPhase.getBoundaryInfo(mpInlets,mpOutlets,resSize = 0)


    #### SET THESE PARAMETERS
    interval = 0.9  ##rate at which to change radius (each loop: radius = interval * radius)
    minSetSize = 10  ##in voxels, remove all smaller w phase, set to 0 for no removal 
    sW = [0.77] ## saturation target list

    drainL = PMMoTo.multiPhase.calcOpenSW(sW,twoPhase,interval,minSetSize)

    # twoPhase.getBoundaryInfo(mpInlets,mpOutlets,resSize = 1)
    # #Initialize from previous fluid distribution
    # inputFile = 'dataOut/Open/twoPhase_open_sw_0.5'
    # twoPhase.initializeMPGrid(inputFile = inputFile) 
    
    # drainL = PMMoTo.multiPhase.calcOpenSW(sW,twoPhase,interval,minSetSize)


    endTime = time.time()
    print("Parallel Time:",endTime-startTime)


if __name__ == "__main__":
    my_function()
    MPI.Finalize()
