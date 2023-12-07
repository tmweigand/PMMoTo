import numpy as np
from mpi4py import MPI
import time
import PMMoTo
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import cProfile

def profile(filename=None, comm = MPI.COMM_WORLD):
  def prof_decorator(f):
    def wrap_f(*args, **kwargs):
      pr = cProfile.Profile()
      pr.enable()
      result = f(*args, **kwargs)
      pr.disable()

      if filename is None:
        pr.print_stats()
      else:
        filename_r = '{}_size_{}_id_{}.out'.format(filename,comm.size,comm.rank)
        pr.dump_stats(filename_r)

      return result
    return wrap_f
  return prof_decorator

@profile(filename="profile_out")
def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subDomains = [2,2,2] # Specifies how Domain is broken among rrocs
    nodes = [100,100,100] # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[2,2],[2,2],[0,0]] # 0: Nothing Assumed  1: Walls 2: Periodic
    dataReadBoundaries = [[2,2],[2,2],[0,0]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[1,0]]
    outlet = [[0,0],[0,0],[0,1]]


    rLookupFile = './rLookups/PA.rLookup'
    file = './testDomains/single.out'


    boundaryLims = [[None,None],
                    [None,None],
                    [None,None]]

    startTime = time.time()
    dataReadkwargs = {'rLookupFile':rLookupFile,
                  'boundaryLims':boundaryLims,
                  'boundaries':dataReadBoundaries,
                  # 'waterMolecule':True,
                  'nodes':nodes}

    domain,sDL,pML = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,"SphereVerlet",file,PMMoTo.readPorousMediaLammpsDump,dataReadkwargs)

    sD_EDT = PMMoTo.calcEDT(sDL,pML.grid,stats = True,sendClass=True)
    PMMoTo.saveGridData("single_old/grid",rank,domain,sDL,pML.grid,dist=sD_EDT.EDT)
    # waterRadius = 0.0
    # sD_twoPhase = np.where(sD_EDT.EDT > waterRadius,2,pML.grid).astype(np.uint8)
    # #sD_twoPhase = PMMoTo.morph(2,sD_twoPhaseMID,sDL,waterRadius)

    # wSets = PMMoTo.sets.collect_sets(sD_twoPhase,2,pML.inlet,pML.outlet,pML.loopInfo,pML.subDomain)
    # connectedFlag = False
    # for s in wSets.sets:
    #    if s.inlet and s.outlet:
    #       connectedFlag=True
    # if connectedFlag and (rank == 0):
    #    print("Connected")
    endTime = time.time()
    print("Parallel Time:",endTime-startTime)

    # setSaveDict = {'inlet': 'inlet',
    #             'outlet':'outlet',
    #             'globalID': 'globalID'}
        
    # PMMoTo.saveSetData("dataOut/set",sDL,wSets,**setSaveDict)



if __name__ == "__main__":
    my_function()
    MPI.Finalize()
