import numpy as np
from mpi4py import MPI
from scipy.ndimage import distance_transform_edt
import edt
import time
import PMMoTo
from skimage.morphology import skeletonize
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import cProfile

def profile(filename=None, comm=MPI.COMM_WORLD):
  def prof_decorator(f):
    def wrap_f(*args, **kwargs):
      pr = cProfile.Profile()
      pr.enable()
      result = f(*args, **kwargs)
      pr.disable()

      if filename is None:
        pr.print_stats()
      else:
        filename_r = filename + ".{}".format(comm.rank)
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
    nodes = [400,400,400] # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[0,0],[1,1],[1,1]] # 0: Nothing Assumed  1: Walls 2: Periodic
    #boundaries = [[1,1],[1,1],[1,1]] # 0: Nothing Assumed  1: Walls 2: Periodic
    #boundaries = [[2,2],[2,2],[2,2]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[1,0],[0,0],[0,0]]
    outlet = [[0,1],[0,0],[0,0]]

    subDomains = [2,2,2] # Specifies how Domain is broken among rrocs
    #nodes = [125,125,100] # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[2,2],[2,2],[0,0]] # 0: Nothing Assumed  1: Walls 2: Periodic
    dataReadBoundaries = [[2,2],[2,2],[0,0]] # 0: Nothing Assumed  1: Walls 2: Periodic
    #boundaries = [[1,1],[1,1],[1,1]] # 0: Nothing Assumed  1: Walls 2: Periodic
    #boundaries = [[2,2],[2,2],[2,2]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[1,0]]
    outlet = [[0,0],[0,0],[0,1]]

    # rLookupFile = './rLookups/PA.rLookup'
    # rLookupFile = None
    file = './testDomains/50pack.out'
    # file = './testDomains/membrane.dump.gz'
    # file = './testDomains/pack_sub.dump.gz'
    #domainFile = open('kelseySpherePackTests/pack_res.out', 'r')

    numSubDomains = np.prod(subDomains)

    drain = False
    testSerial = False
    testAlgo = False

    pC = [140,160]
    cutoffs = [0]
    startTime = time.time()

    # domain,sDL = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,"Sphere",file,PMMoTo.readPorousMediaLammpsDump,rLookupFile)
    domain,sDL,pML = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,"Sphere",file,PMMoTo.readPorousMediaXYZR)

    numFluidPhases = 2

    twoPhase = PMMoTo.multiPhase.multiPhase(pML,numFluidPhases)

    wRes  = [[0,0],[0,0],[0,0]]
    nwRes = [[0,0],[0,0],[0,0]]
    mpInlets = {twoPhase.wID:wRes,twoPhase.nwID:nwRes}

    wOut  = [[0,0],[0,0],[0,0]]
    nwOut = [[0,0],[0,0],[0,0]]
    mpOutlets = {twoPhase.wID:wOut,twoPhase.nwID:nwOut}

    twoPhase.initializeMPGrid(constantPhase = twoPhase.wID)
    twoPhase.getBoundaryInfo(mpInlets,mpOutlets,resSize=0)


    sD_EDT = PMMoTo.calcEDT(sDL,pML.grid,stats = True,sendClass=True)

    
    if drain:
        drainL = PMMoTo.multiPhase.calcDrainage(pC,twoPhase)

    #rad = 0.1
    #sDMorphL = PMMoTo.morph(rank,size,domain,sDL,sDL.grid,rad)

    #sDMSL = PMMoTo.medialAxis.medialSurfaceEval(rank,size,domain,sDL,sDL.grid)

    sDMAL = PMMoTo.medialAxis.medialAxisEval(sDL,pML,pML.grid,sD_EDT.EDT,connect = True, trim  = True)
    #sDMAL = PMMoTo.medialAxis.medialAxisTrim(sDMAL,pML,sDL,sD_EDT.EDT,cutoffs)


    endTime = time.time()
    print("Parallel Time:",endTime-startTime)

    procID = rank*np.ones_like(pML.grid)

    ### Save Grid Data where kwargs are used for saving other grid data (i.e. EDT, Medial Axis)
    PMMoTo.saveGridData("dataOut/grid",rank,domain,sDL,pML.grid,dist=sD_EDT.EDT,MA=sDMAL.MA,PROC=procID)#,ind = drainL.ind, nwp=drainL.nwp,nwpFinal=drainL.nwpFinal)

    ### Save Set Data from Medial Axis
    ### kwargs include any attribute of Set class (see sets.pyx)

    # setSaveDict = {'inlet': 'inlet',
    #             'outlet':'outlet',
    #             'boundary': 'boundary',
    #             'localID': 'localID',
    #             }

    for s in sDMAL.Sets.sets:
        s.numConnectedSets = len(s.connectedSets)

    setSaveDict = {'inlet': 'inlet',
                'outlet':'outlet',
                'trim' :'trim',
                'inaccessible' :'inaccessible',
                'boundary': 'boundary',
                'localID': 'localID',
                'type': 'type',
                'numBoundaries': 'numBoundaries',
                'pathID': 'pathID',
                'numConnectedSets':'numConnectedSets'}
    
    #PMMoTo.saveSetData("dataOut/set",sDL,drainL,**setSaveDict)
    
    PMMoTo.saveSetData("dataOut/set",sDL,sDMAL.Sets,**setSaveDict)

    if testSerial:

        if rank == 0:
            sD = np.empty((numSubDomains), dtype = object)
            sDEDT = np.empty((numSubDomains), dtype = object)
            pM = np.empty((numSubDomains), dtype = object)
            if drain:
                sDDrain = np.empty((numSubDomains), dtype = object)
            sDMA = np.empty((numSubDomains), dtype = object)
            sD[0] = sDL
            sDEDT[0] = sD_EDT
            pM[0] = pML
            if drain:
                sDDrain[0] = drainL
            sDMA[0] = sDMAL
            for neigh in range(1,numSubDomains):
                sD[neigh] = comm.recv(source=neigh)
                sDEDT[neigh] = comm.recv(source=neigh)
                pM[neigh] = comm.recv(source=neigh)
                if drain:
                    sDDrain[neigh] = comm.recv(source=neigh)
                sDMA[neigh] = comm.recv(source=neigh)

        if rank > 0:
            for neigh in range(1,numSubDomains):
                if rank == neigh:
                    comm.send(sDL,dest=0)
                    comm.send(sD_EDT,dest=0)
                    comm.send(pML,dest=0)
                    if drain:
                        comm.send(drainL,dest=0)
                    comm.send(sDMAL,dest=0)


        if rank==0:
            if testAlgo:
                
                startTime = time.time()

                _,sphereData = PMMoTo.readPorousMediaXYZR(file)
                # _,sphereData = PMMoTo.readPorousMediaLammpsDump(file,rLookupFile)
                
                ##### To GENERATE SINGLE PROC TEST CASE ######
                x = np.linspace(domain.domainSize[0,0]+domain.dX/2, domain.domainSize[0,1]-domain.dX/2, nodes[0])
                y = np.linspace(domain.domainSize[1,0]+domain.dY/2, domain.domainSize[1,1]-domain.dY/2, nodes[1])
                z = np.linspace(domain.domainSize[2,0]+domain.dZ/2, domain.domainSize[2,1]-domain.dZ/2, nodes[2])
                gridOut = PMMoTo.domainGen(x,y,z,sphereData)
                gridOut = np.asarray(gridOut)

                pG = [0,0,0]
                pgSize = int(nodes[0]/2)

                if boundaries[0][0] == 1:
                    pG[0] = 1
                if boundaries[1][0] == 1:
                    pG[1] = 1
                if boundaries[2][0] == 1:
                    pG[2] = 1

                periodic = [False,False,False]
                if boundaries[0][0] == 2:
                    periodic[0] = True
                    pG[0] = pgSize
                if boundaries[1][0] == 2:
                    periodic[1] = True
                    pG[1] = pgSize
                if boundaries[2][0] == 2:
                    periodic[2] = True
                    pG[2] = pgSize

                gridOut = np.pad (gridOut, ((pG[0], pG[0]), (pG[1], pG[1]), (pG[2], pG[2])), 'wrap')

                if boundaries[0][0] == 1:
                    gridOut[0,:,:] = 0
                if boundaries[0][1] == 1:
                    gridOut[-1,:,:] = 0
                if boundaries[1][0] == 1:
                    gridOut[:,0,:] = 0
                if boundaries[1][1] == 1:
                    gridOut[:,-1,:] = 0
                if boundaries[2][0] == 1:
                    gridOut[:,:,0] = 0
                if boundaries[2][1] == 1:
                    gridOut[:,:,-1] = 0


                realDT = edt.edt3d(gridOut, anisotropy=(domain.dX, domain.dY, domain.dZ))
                edtV,_ = distance_transform_edt(gridOut,sampling=[domain.dX, domain.dY, domain.dZ],return_indices=True)
                skelMA = skeletonize(gridOut)
                gridCopy = np.copy(gridOut)
                #realMA,_ = PMMoTo.medialAxis.medialAxis._compute_thin_image_test(gridCopy)
                #realMA = PMMoTo.medialAxis.medialAxis._compute_thin_image(gridCopy)
                endTime = time.time()

                print("Serial Time:",endTime-startTime)

                arrayList = [gridOut,realDT,edtV,skelMA]

                for i,arr in enumerate(arrayList):
                    if pG[0] > 0 and pG[1]==0 and pG[2]==0:
                        arrayList[i] = arr[pG[0]:-pG[0],:,:]
                    elif pG[0]==0 and pG[1] > 0 and pG[2]==0:
                        arrayList[i] = arr[:,pG[1]:-pG[1],:]
                    elif pG[0]==0 and pG[1]==0 and pG[2] > 0:
                        arrayList[i] = arr[:,:,pG[2]:-pG[2]]
                    elif pG[0] > 0 and pG[1]==0 and pG[2] > 0:
                        arrayList[i] = arr[pG[0]:-pG[0],:,pG[2]:-pG[2]]
                    elif pG[0] > 0 and pG[1] > 0 and pG[2]==0:
                        arrayList[i] = arr[pG[0]:-pG[0],pG[1]:-pG[1],:]
                    elif pG[0]==0 and pG[1] > 0 and pG[2] > 0:
                        arrayList[i] = arr[:,pG[1]:-pG[1],pG[2]:-pG[2]]
                    elif pG[0] > 0 and pG[1] > 0 and pG[2] > 0:
                        arrayList[i] = arr[pG[0]:-pG[0],pG[1]:-pG[1],pG[2]:-pG[2]]
                ####################################################################

                gridOut = arrayList[0]
                realDT = arrayList[1]
                edtV = arrayList[2]
                #realMA = arrayList[3]
                skelMA = arrayList[3]


                ### Reconstruct SubDomains to Check EDT ####
                checkGrid = np.zeros_like(realDT)
                n = 0
                for i in range(0,subDomains[0]):
                    for j in range(0,subDomains[1]):
                        for k in range(0,subDomains[2]):
                            checkGrid[sD[n].indexStart[0]+sD[n].buffer[0]: sD[n].indexStart[0]+sD[n].nodes[0]-sD[n].buffer[1],
                                      sD[n].indexStart[1]+sD[n].buffer[2]: sD[n].indexStart[1]+sD[n].nodes[1]-sD[n].buffer[3],
                                      sD[n].indexStart[2]+sD[n].buffer[4]: sD[n].indexStart[2]+sD[n].nodes[2]-sD[n].buffer[5]] \
                                      = pM[n].grid[sD[n].buffer[0] : pM[n].grid.shape[0] - sD[n].buffer[1],
                                                      sD[n].buffer[2] : pM[n].grid.shape[1] - sD[n].buffer[3],
                                                      sD[n].buffer[4] : pM[n].grid.shape[2] - sD[n].buffer[5]]



                            n = n + 1

                diffGrid = gridOut-checkGrid
                print("L2 Grid Error Norm",np.linalg.norm(diffGrid) )


                checkEDT = np.zeros_like(realDT)
                n = 0
                for i in range(0,subDomains[0]):
                    for j in range(0,subDomains[1]):
                        for k in range(0,subDomains[2]):
                            checkEDT[sD[n].indexStart[0]+sD[n].buffer[0]: sD[n].indexStart[0]+sD[n].nodes[0]-sD[n].buffer[1],
                                     sD[n].indexStart[1]+sD[n].buffer[2]: sD[n].indexStart[1]+sD[n].nodes[1]-sD[n].buffer[3],
                                     sD[n].indexStart[2]+sD[n].buffer[4]: sD[n].indexStart[2]+sD[n].nodes[2]-sD[n].buffer[5]] \
                                     = sDEDT[n].EDT[sD[n].buffer[0] : pM[n].grid.shape[0] - sD[n].buffer[1],
                                                    sD[n].buffer[2] : pM[n].grid.shape[1] - sD[n].buffer[3],
                                                    sD[n].buffer[4] : pM[n].grid.shape[2] - sD[n].buffer[5]]
                            n = n + 1

                diffEDT = np.abs(realDT-checkEDT)
                diffEDT2 = np.abs(edtV-checkEDT)

                print("L2 EDT Error Norm",np.linalg.norm(diffEDT) )
                print("L2 EDT Error Norm 2",np.linalg.norm(diffEDT2) )
                print("L2 EDT Error Norm 2",np.linalg.norm(realDT-edtV) )

                print("LI EDT Error Norm",np.max(diffEDT) )
                print("LI EDT Error Norm 2",np.max(diffEDT2) )
                print("LI EDT Error Norm 2",np.max(realDT-edtV) )

                checkMA = np.zeros_like(realDT)
                n = 0
                for i in range(0,subDomains[0]):
                    for j in range(0,subDomains[1]):
                        for k in range(0,subDomains[2]):
                            checkMA[sD[n].indexStart[0]+sD[n].buffer[0]: sD[n].indexStart[0]+sD[n].nodes[0]-sD[n].buffer[1],
                                     sD[n].indexStart[1]+sD[n].buffer[2]: sD[n].indexStart[1]+sD[n].nodes[1]-sD[n].buffer[3],
                                     sD[n].indexStart[2]+sD[n].buffer[4]: sD[n].indexStart[2]+sD[n].nodes[2]-sD[n].buffer[5]] \
                                     = sDMA[n].MA[sD[n].buffer[0] : pM[n].grid.shape[0] - sD[n].buffer[1],
                                                    sD[n].buffer[2] : pM[n].grid.shape[1] - sD[n].buffer[3],
                                                    sD[n].buffer[4] : pM[n].grid.shape[2] - sD[n].buffer[5]]
                            n = n + 1

                print("MA Check Sum:",np.sum(checkMA),np.sum(skelMA))
                #diffMA = np.abs(realMA-checkMA)
                #print("L2 MA Error Total Different Voxels",np.sum(diffMA) )

                #PMMoTo.saveGridOneProc("dataOut/diffMA",x,y,z,diffMA)
                #PMMoTo.saveGridOneProc("dataOut/trueMA",x,y,z,np.ascontiguousarray(realMA))
                PMMoTo.saveGridOneProc("dataOut/skelMA",x,y,z,np.ascontiguousarray(skelMA))



if __name__ == "__main__":
    my_function()
    MPI.Finalize()
