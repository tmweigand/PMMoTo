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
    #nodes = [140,30,30]  ##res = 10
    #nodes = [280,60,60]   ##res = 20
    #nodes = [560,120,120] ##res = 40
    nodes = [1120,240,240] ##res = 80
    # nodes = [1680,360,360] ##res = 120

    ##half voxel used for 10 degree CA and less
    half_voxel = 0.05  ##140: 0.05, 280: 0.025, 560: 0.0125
    res = 1

    boundaries = [[0,0],[0,0],[0,0]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[0,0]]
    outlet = [[0,0],[0,0],[0,0]]
    
    ############################################################################################################################################
    ### get CAgrid for LVCA/Liu method
    # domain,sDL,pML = PMMoTo.genDomainSubDomainCA(rank,size,subDomains,nodes,boundaries,inlet,outlet,"InkBottle",None,None)
    # numFluidPhases = 2
    # twoPhase = PMMoTo.multiPhase.multiPhase(pML,numFluidPhases)

    # wRes  = [[1,0],[0,0],[0,0]]
    # nwRes = [[0,1],[0,0],[0,0]]
    # mpInlets = {twoPhase.wID:wRes,twoPhase.nwID:nwRes}

    # wOut  = [[0,0],[0,0],[0,0]]
    # nwOut = [[0,0],[0,0],[0,0]]
    # mpOutlets = {twoPhase.wID:wOut,twoPhase.nwID:nwOut}
    
    # #Initialize wetting saturated somain
    # twoPhase.initializeMPGrid(constantPhase = twoPhase.wID) 
    # twoPhase.getBoundaryInfo(mpInlets,mpOutlets,resSize = res)
    
    # CAgrid = np.copy(twoPhase.porousMedia.grid)
    ###############################################################################################################################################
    
    # domain,sDL,pML = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,"InkBottle",None,None)
    domain,sDL,pML = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,"CapTube",None,None)

    numFluidPhases = 2
    twoPhase = PMMoTo.multiPhase.multiPhase(pML,numFluidPhases)

    wRes  = [[1,0],[0,0],[0,0]]
    nwRes = [[0,1],[0,0],[0,0]]
    mpInlets = {twoPhase.wID:wRes,twoPhase.nwID:nwRes}

    wOut  = [[0,0],[0,0],[0,0]]
    nwOut = [[0,0],[0,0],[0,0]]
    mpOutlets = {twoPhase.wID:wOut,twoPhase.nwID:nwOut}
    
    #Initialize wetting saturated domain
    ##res size should be 1 for imbibition
    twoPhase.initializeMPGrid(constantPhase = twoPhase.wID) ##drainage
    # twoPhase.initializeMPGrid(constantPhase = twoPhase.nwID) ##imbibition
    
    # #Initialize from previous fluid distribution
    # inputFile = 'dataOut/twoPhase/twoPhase_imbibe_pc_1.61783'
    # twoPhase.initializeMPGrid(inputFile = inputFile) 
    

    
    CA = 20
#     pC = [1.45662, 1.47303, 1.52342, 1.58833, 1.66993, 1.77076, 1.89382, 2.04261, 2.22115,
# 2.4339, 2.68565, 2.98117, 3.32459, 3.71849, 4.16236, 4.65074, 5.17101, 5.7015, 6.21098,
# 6.6607, 7.00993, 7.22414, 7.23008]

    pC = [1.43]
    
    # Drainage
    twoPhase.getBoundaryInfo(mpInlets,mpOutlets,resSize = res)
    # drainL = PMMoTo.multiPhase.calcDrainageCA(pC,twoPhase,CAgrid,CA)
    rainL = PMMoTo.multiPhase.calcDrainageSchulzCA(pC,twoPhase,CA)
    
    # # #Initialize from previous fluid distribution
    # inputFile = 'dataOut/twoPhase/twoPhase_drain_pc_7.69393'
    # twoPhase.initializeMPGrid(inputFile = inputFile) 

    # CA = 50
    # pC = [2.86704]
    
    # Imbibition
    # twoPhase.getBoundaryInfo(mpInlets,mpOutlets,resSize = res)
    
    # drainL = PMMoTo.multiPhase.calcImbibitionCA(pC,twoPhase,CAgrid,CA,half_voxel)
    # #drainL = PMMoTo.multiPhase.calcImbibition(pC,twoPhase)
    

if __name__ == "__main__":
    my_function()
