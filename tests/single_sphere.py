import numpy as np
from mpi4py import MPI
from scipy.ndimage import distance_transform_edt
import os
import edt
import time
import PMMoTo


def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subDomains = [2,2,2]
    nodes = [300,300,300]
    boundaries = [2,2,2]
    inlet  = [1,0,0]
    outlet = [-1,0,0]
    rLookupFile = './rLookups/single.rLookup'
    # rLookupFile = None
    # file = './testDomains/50pack.out'
    file = './testDomains/sphere.dump'
    # file = './testDomains/pack_sub.dump.gz'
    #domainFile = open('kelseySpherePackTests/pack_res.out', 'r')
    res = 1 ### Assume that the reservoir is always at the inlet!
    boundaryLims = [[None, None],
                    [None, None],
                    [None, None]]
    startTime = time.time()
    # optionally be able to set location of domain boundaries
    dataReadkwargs = {'rLookupFile':rLookupFile,
                      'boundaryLims':boundaryLims}
    domain,sDL = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,res,"SphereVerlet",file,PMMoTo.readPorousMediaLammpsDump,dataReadkwargs)

    sDEDTL = PMMoTo.calcEDT(rank,size,domain,sDL,sDL.grid,stats = True)
    MF = PMMoTo.minkowskiEval(sDEDTL.EDT,res=[domain.dX,domain.dY,domain.dZ])
    MF_gathered = comm.gather(MF,root=0)
    if rank == 0:
        max_length = 0
        for MF in MF_gathered:
            if len(MF[0]) > max_length:
                max_length = len(MF[0])
                dist = MF[0]
        
        volume = np.zeros_like(dist)
        surface = np.zeros_like(dist)
        curvature = np.zeros_like(dist)
        euler = np.zeros_like(dist)

        for MF in MF_gathered:
            for i in range(min(len(dist),len(MF[0]))):
                volume[i] += MF[1][i]
                surface[i] += MF[2][i]
                curvature[i] += MF[3][i]
                euler[i] += MF[4][i]

        # print(dist)
        # print(volume)
        # print(surface)
        # print(curvature)
        # print(euler)
        np.savetxt('MF.csv',np.array([dist,volume,surface,curvature,euler]),delimiter=',')

        

    cutoffs = [0]

    rad = 0.1
    # sDMorphL = PMMoTo.morph(rank,size,domain,sDL,sDL.grid,rad)

    # sDMAL = PMMoTo.medialAxis.medialAxisEval(rank,size,domain,sDL,sDL.grid,sDEDTL.EDT,connect = False,cutoffs = cutoffs)


    endTime = time.time()
    print("Parallel Time:",endTime-startTime)

    # PMMoTo.saveGridData("paDataOut/grid",rank,domain,sDL, dist=sDEDTL.EDT,)
    # ### Save Grid Data where kwargs are used for saving other grid data (i.e. EDT, Medial Axis)
    PMMoTo.saveGridData("ssDataOut/grid",rank,domain,sDL, dist=sDEDTL.EDT)

    # ### Save Set Data from Medial Axis
    # ### kwargs include any attribute of Set class (see sets.pyx)

    # setSaveDict = {'inlet': 'inlet',
    #             'outlet':'outlet',
    #             'trim' :'trim',
    #             'inaccessible':'inaccessible',
    #             'globalID':'globalID',
    #             'boundary': 'boundary',
    #             'localID': 'localID',
    #             'type': 'type',
    #             'numBoundaries': 'numBoundaries',
    #             'pathID':'pathID',
    #             'globalPathIDs':'globalPathIDs'}
    
    # PMMoTo.saveSetData("padataOut/set",rank,domain,sDL,sDMAL,**setSaveDict)

if __name__ == "__main__":
    my_function()
    MPI.Finalize()