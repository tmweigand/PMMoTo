import numpy as np
from mpi4py import MPI
from scipy.ndimage import distance_transform_edt
import edt
import time
import PMMoTo
from skimage.morphology import skeletonize
import os
import re
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import cProfile

# def profile(filename=None, comm = MPI.COMM_WORLD):
#   def prof_decorator(f):
#     def wrap_f(*args, **kwargs):
#       pr = cProfile.Profile()
#       pr.enable()
#       result = f(*args, **kwargs)
#       pr.disable()

#       if filename is None:
#         pr.print_stats()
#       else:
#         filename_r = '{}_size_{}_id_{}.out'.format(filename,comm.size,comm.rank)
#         pr.dump_stats(filename_r)

#       return result
#     return wrap_f
#   return prof_decorator

# @profile(filename="profile_out")
def my_function():
    setSaveDict = {'inlet': 'inlet',
            'outlet':'outlet',
            'globalID': 'globalID'}


    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subDomains = [4,4,8] # Specifies how Domain is broken among rrocs
    nodes = [875,875,750] # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[2,2],[2,2],[0,0]] # 0: Nothing Assumed  1: Walls 2: Periodic
    dataReadBoundaries = [[2,2],[2,2],[0,0]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[1,0]]
    outlet = [[0,0],[0,0],[0,1]]


    rLookupFile = './rLookups/PA.rLookup'
    averagingDirectory = '../../RV-P3/64x/03_90/perm_v2/membrane/'

    files = [averagingDirectory + file for file in os.listdir(averagingDirectory)]
    files = np.array(sorted(files, key=lambda x: (int(re.sub(r'[^0-9]', '', x)))))[-8000:]
    nWindows = 1
    boundaryLims = [[None,None],
                    [None,None],
                    [-75.329262, 75.329262]]

    dataReadkwargs = {'rLookupFile':rLookupFile,
                  'boundaryLims':boundaryLims,
                  'boundaries':dataReadBoundaries,
                  'waterMolecule':True,
                  'nodes':nodes}
    dims = []
    splitFiles = np.split(files,nWindows)
    waterRadii = [0.8,0.9,1.0,1.1,1.2,1.3,1.375]
    maxRadius = 0.9
    for j in range(nWindows):
        for i,file in enumerate(splitFiles[j]):
            timestep=file.split('/')[-1].split('.')[1]
            if int(timestep) < int(sys.argv[1]):
                continue
            startTime = time.time()
            domain,sDL,pML = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,"SphereVerlet",file,PMMoTo.readPorousMediaLammpsDump,dataReadkwargs)
            res=np.array([round(domain.dX,10),round(domain.dY,10),round(domain.dZ,10)])          
            if rank == 0:
                print(file)
                print(res)
            sD_EDT = PMMoTo.calcEDT(sDL,pML.grid,stats = True,sendClass=True)


            for jj,waterRadius in enumerate(waterRadii):
              if rank == 0:
                 print("Testing radius {}".format(waterRadius))
              sD_twoPhase = np.where(sD_EDT.EDT > waterRadius,2,pML.grid).astype(np.uint8)
              wSets = PMMoTo.sets.collect_sets(sD_twoPhase,2,pML.inlet,pML.outlet,pML.loopInfo,pML.subDomain)
              connectedFlag = False
              for s in wSets.sets:
                  if s.inlet and s.outlet:
                      connectedFlag = True
                      # PMMoTo.saveSetData("dataOut_{}/set".format(str(timestep)),sDL,wSets,**setSaveDict)
              if rank==0:  
                  if connectedFlag:
                    print("Timestep {0} had a connection with test radius of {1}".format(timestep,waterRadius))
                    maxRadius = waterRadius
            if rank == 0:
               print(waterRadii)
               print("Max radius: ", maxRadius)
            endTime = time.time()
            if rank == 0:
                print("That file took: {} seconds".format(endTime-startTime))

if __name__ == "__main__":
    my_function()
    MPI.Finalize()
