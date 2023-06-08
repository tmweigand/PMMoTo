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
    nodes = [500,500,500]
    boundaries = [[0,0],[0,0],[0,0]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[1,0]]
    outlet = [[0,0],[0,0],[0,1]]
    rLookupFile = './rLookups/PA.rLookup'
    # rLookupFile = None
    # file = './testDomains/50pack.out'
    file = './testDomains/membranedata.71005000.gz'
    # file = './testDomains/pack_sub.dump.gz'
    #domainFile = open('kelseySpherePackTests/pack_res.out', 'r')
    #[-87.8841396550089, 87.8841396550089]
    numSubDomains = np.prod(subDomains)
    boundaryLims = [[50, 99.82648613],
                    [0,49.82648613],
                    [0,49.82648613]]
    startTime = time.time()
    # optionally be able to set location of domain boundaries
    dataReadkwargs = {'rLookupFile':rLookupFile,
                      'boundaryLims':boundaryLims}
    
    domain,sDL,pML = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,"SphereVerlet",file,PMMoTo.readPorousMediaLammpsDump,dataReadkwargs)
    res=np.array([round(domain.dX,10),round(domain.dY,10),round(domain.dZ,10)])
    sD_EDT = PMMoTo.calcEDT(sDL,pML.grid,stats = True,sendClass=True)
    MF = PMMoTo.minkowskiEval(sD_EDT.EDT,res=res)
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

        np.savetxt('MF.csv',np.array([dist,volume,surface,curvature,euler]),delimiter=',')
#     cutoffs = [0]

#     rad = 0.1

    sDMAL = PMMoTo.medialAxis.medialAxisEval(sDL,pML,sD_EDT.EDT,connect = False,cutoffs=[0,1])


    endTime = time.time()
    print("Parallel Time:",endTime-startTime)

    procID = rank*np.ones_like(pML.grid)

#     # PMMoTo.saveGridData("paDataOut/grid",rank,domain,sDL, dist=sDEDTL.EDT,)
#     # ### Save Grid Data where kwargs are used for saving other grid data (i.e. EDT, Medial Axis)
    PMMoTo.saveGridData("dataOut/grid",rank,domain,sDL,pML.grid,dist=sD_EDT.EDT,MA=sDMAL.MA,PROC=procID)

#     ### Save Set Data from Medial Axis
#     ### kwargs include any attribute of Set class (see sets.pyx)

#     setSaveDict = {'inlet': 'inlet',
#                 'outlet':'outlet',
#                 'boundary': 'boundary',
#                 'localID': 'localID'}

#     setSaveDict = {'inlet': 'inlet',
#                 'outlet':'outlet',
#                 'trim' :'trim',
#                 'boundary': 'boundary',
#                 'localID': 'localID',
#                 'type': 'type',
#                 'numBoundaries': 'numBoundaries',
#                 'globalPathID':'globalPathID'}
    
#     #PMMoTo.saveSetData("dataOut/set",sDL,drainL,**setSaveDict)
    
#     #PMMoTo.saveSetData("dataOut/set",sDL,sDMAL,**setSaveDict)

if __name__ == "__main__":
    my_function()
    MPI.Finalize()