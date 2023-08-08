import numpy as np
from mpi4py import MPI
import time
import PMMoTo


def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subDomains = [2,2,2] # Specifies how Domain is broken among rrocs
    nodes = [200,200,200] # Total Number of Nodes in Domain

    ## 200 = original
    
    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[2,2],[2,2],[0,0]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[0,0]]
    outlet = [[0,0],[0,0],[0,0]]


    file = './testDomains/50pack.out'

    #pC = [140,160]

    startTime = time.time()
    domain,sDL,pML = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,"Sphere",file,PMMoTo.readPorousMediaXYZR)

    numFluidPhases = 2
    twoPhase = PMMoTo.multiPhase.multiPhase(pML,numFluidPhases)

    wRes  = [[0,0],[0,0],[1,0]]
    nwRes = [[0,0],[0,0],[0,1]]
    mpInlets = {twoPhase.wID:wRes,twoPhase.nwID:nwRes}

    wOut  = [[0,0],[0,0],[0,0]]
    nwOut = [[0,0],[0,0],[0,0]]
    mpOutlets = {twoPhase.wID:wOut,twoPhase.nwID:nwOut}
    
    #Initialize wetting saturated somain
    twoPhase.initializeMPGrid(constantPhase = twoPhase.wID) 
    
    ### MUST INCLUDE RESERVOIR > 0 FOR DRAINAGE (open does not need one)
    # twoPhase.getBoundaryInfo(mpInlets,mpOutlets,resSize = 10)



    # pC = [50,100,150,200]
    # twoPhase.getBoundaryInfo(mpInlets,mpOutlets,resSize = 10)
    #interval = 0.9
    minSetSize = 2  ##voxels (for morph open only: remove all smaller w phase, set to 0 for no removal)
    #sW = [0.95,0.85,0.75,0.65,0.55,0.45,0.35,0.25,0.15,0.05]
    
    print("DRAIN")
    pC = [50,100,110,120,130,140,150,160,170,180,190,200,250]
    twoPhase.getBoundaryInfo(mpInlets,mpOutlets,resSize = 10)
    drainL = PMMoTo.multiPhase.calcDrainage(pC,twoPhase,minSetSize)
    
    #drainL = PMMoTo.multiPhase.calcDrainageSW(sW,twoPhase,interval)
    #drainL = PMMoTo.multiPhase.calcOpenSW(sW,twoPhase,interval,minSetSize)
    #drainL = PMMoTo.multiPhase.calcImbibition(pC,twoPhase)
    
    #Initialize from previous fluid distribution
    inputFile = 'dataOut/twoPhase/twoPhase_drain_pc_250'
    twoPhase.initializeMPGrid(inputFile = inputFile) 
    
    print("IMBIBE")
    pC = [10,20,30,40,50,60,70,80,90,100,103,106,110,200]
    twoPhase.getBoundaryInfo(mpInlets,mpOutlets,resSize = 1)
    drainL = PMMoTo.multiPhase.calcImbibition(pC,twoPhase)
    
    
    #Initialize from previous fluid distribution
    inputFile = 'dataOut/twoPhase/twoPhase_imbibe_pc_10'
    twoPhase.initializeMPGrid(inputFile = inputFile) 
    
    
    print("DRAIN")
    pC = [51,101,111,121,131,141,151,161,171,181,191,201,251]
    twoPhase.getBoundaryInfo(mpInlets,mpOutlets,resSize = 10)
    drainL = PMMoTo.multiPhase.calcDrainage(pC,twoPhase,minSetSize)
    
    

    

    endTime = time.time()
    print("Parallel Time:",endTime-startTime)


if __name__ == "__main__":
    my_function()
    MPI.Finalize()
