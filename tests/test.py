import numpy as np
from mpi4py import MPI
from scipy.ndimage import distance_transform_edt
import os
import edt
import time
import PMMoTo

import cProfile

# def profile(filename=None, comm=MPI.COMM_WORLD):
#   def prof_decorator(f):
#     def wrap_f(*args, **kwargs):
#       pr = cProfile.Profile()
#       pr.enable()
#       result = f(*args, **kwargs)
#       pr.disable()

#       if filename is None:
#         pr.print_stats()
#       else:
#         filename_r = filename + ".{}".format(comm.rank)
#         pr.dump_stats(filename_r)

#       return result
#     return wrap_f
#   return prof_decorator

# @profile(filename="profile_out")
def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subDomains = [2,2,2]
    nodes = [351,351,351]
    boundaries = [2,2,2]
    inlet  = [1,0,0]
    outlet = [-1,0,0]
    # rLookupFile = './rLookups/PA.rLookup'
    # rLookupFile = None
    file = './testDomains/10pack.out'
    # file = './testDomains/membrane.dump.gz'
    # file = './testDomains/pack_sub.dump.gz'
    #domainFile = open('kelseySpherePackTests/pack_res.out', 'r')
    res = 1 ### Assume that the reservoir is always at the inlet!

    numSubDomains = np.prod(subDomains)

    drain = False
    testSerial = False
    testAlgo = False

    pC = [143]

    startTime = time.time()

    # domain,sDL = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,res,"Sphere",file,PMMoTo.readPorousMediaLammpsDump,rLookupFile)
    domain,sDL = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,res,"Sphere",file,PMMoTo.readPorousMediaXYZR)

    sDEDTL = PMMoTo.calcEDT(rank,size,domain,sDL,sDL.grid,stats = True)

    cutoff = 0.006
    if drain:
        drainL,_ = PMMoTo.calcDrainage(rank,size,pC,domain,sDL,inlet,sDEDTL)

    rad = 0.1
    sDMorphL = PMMoTo.morph(rank,size,domain,sDL,sDL.grid,rad)

    sDMAL = PMMoTo.medialAxis.medialAxisEval(rank,size,domain,sDL,sDL.grid,sDEDTL.EDT,connect = False,cutoff = cutoff)


    endTime = time.time()
    print("Parallel Time:",endTime-startTime)


    ### Save Grid Data where kwargs are used for saving other grid data (i.e. EDT, Medial Axis)
    PMMoTo.saveGridData("dataOut/gridSurface",rank,domain,sDL, dist=sDEDTL.EDT, MA=sDMAL.MA)

    ### Save Set Data from Medial Axis
    ### kwargs include any attribute of Set class (see sets.pyx)

    setSaveDict = {'inlet': 'inlet',
                'outlet':'outlet',
                'trim' :'trim',
                'boundary': 'boundary',
                'localID': 'localID',
                'type': 'type',
                'numBoundaries': 'numBoundaries',
                'globalPathID':'globalPathID'}
    
    PMMoTo.saveSetData("dataOut/setSurface",rank,domain,sDL,sDMAL,**setSaveDict)


    if testSerial:

        if rank == 0:
            sD = np.empty((numSubDomains), dtype = object)
            sDEDT = np.empty((numSubDomains), dtype = object)
            if drain:
                sDDrain = np.empty((numSubDomains), dtype = object)
            sDMorph = np.empty((numSubDomains), dtype = object)
            sDMA = np.empty((numSubDomains), dtype = object)
            sD[0] = sDL
            sDEDT[0] = sDEDTL
            if drain:
                sDDrain[0] = drainL
            sDMorph[0] = sDMorphL
            sDMA[0] = sDMAL
            for neigh in range(1,numSubDomains):
                sD[neigh] = comm.recv(source=neigh)
                sDEDT[neigh] = comm.recv(source=neigh)
                if drain:
                    sDDrain[neigh] = comm.recv(source=neigh)
                sDMorph[neigh] = comm.recv(source=neigh)
                sDMA[neigh] = comm.recv(source=neigh)

        if rank > 0:
            for neigh in range(1,numSubDomains):
                if rank == neigh:
                    comm.send(sDL,dest=0)
                    comm.send(sDEDTL,dest=0)
                    if drain:
                        comm.send(drainL,dest=0)
                    comm.send(sDMorphL,dest=0)
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
                pgSize = nodes[0]

                if boundaries[0] == 1:
                    pG[0] = 1
                if boundaries[1] == 1:
                    pG[1] = 1
                if boundaries[2] == 1:
                    pG[2] = 1

                periodic = [False,False,False]
                if boundaries[0] == 2:
                    periodic[0] = True
                    pG[0] = pgSize
                if boundaries[1] == 2:
                    periodic[1] = True
                    pG[1] = pgSize
                if boundaries[2] == 2:
                    periodic[2] = True
                    pG[2] = pgSize

                gridOut = np.pad (gridOut, ((pG[0], pG[0]), (pG[1], pG[1]), (pG[2], pG[2])), 'wrap')

                if boundaries[0] == 1:
                    gridOut[0,:,:] = 0
                    gridOut[-1,:,:] = 0
                if boundaries[1] == 1:
                    gridOut[:,0,:] = 0
                    gridOut[:,-1,:] = 0
                if boundaries[2] == 1:
                    gridOut[:,:,0] = 0
                    gridOut[:,:,-1] = 0


                realDT = edt.edt3d(gridOut, anisotropy=(domain.dX, domain.dY, domain.dZ))
                edtV,_ = distance_transform_edt(gridOut,sampling=[domain.dX, domain.dY, domain.dZ],return_indices=True)
                gridCopy = np.copy(gridOut)
                realMA = PMMoTo.medialAxis.skeletonize._compute_thin_image_surface(gridCopy)
                endTime = time.time()

                print("Serial Time:",endTime-startTime)

                if pG[0] > 0 and pG[1]==0 and pG[2]==0:
                    gridOut = gridOut[pG[0]:-pG[0],:,:]
                    realDT = realDT[pG[0]:-pG[0],:,:]
                    edtV = edtV[pG[0]:-pG[0],:,:]
                    realMA = realMA[pG[0]:-pG[0],:,:]

                elif pG[0]==0 and pG[1] > 0 and pG[2]==0:
                    gridOut = gridOut[:,pG[1]:-pG[1],:]
                    realDT = realDT[:,pG[1]:-pG[1],:]
                    edtV = edtV[:,pG[1]:-pG[1],:]
                    realMA = realMA[:,pG[1]:-pG[1],:]

                elif pG[0]==0 and pG[1]==0 and pG[2] > 0:
                    gridOut = gridOut[:,:,pG[2]:-pG[2]]
                    realDT = realDT[:,:,pG[2]:-pG[2]]
                    edtV = edtV[:,:,pG[2]:-pG[2]]
                    realMA = realMA[:,:,pG[2]:-pG[2]]

                elif pG[0] > 0 and pG[1]==0 and pG[2] > 0:
                    gridOut = gridOut[pG[0]:-pG[0],:,pG[2]:-pG[2]]
                    realDT = realDT[pG[0]:-pG[0],:,pG[2]:-pG[2]]
                    edtV = edtV[pG[0]:-pG[0],:,pG[2]:-pG[2]]
                    realMA = realMA[pG[0]:-pG[0],:,pG[2]:-pG[2]]

                elif pG[0] > 0 and pG[1] > 0 and pG[2]==0:
                    gridOut = gridOut[pG[0]:-pG[0],pG[1]:-pG[1],:]
                    realDT = realDT[pG[0]:-pG[0],pG[1]:-pG[1],:]
                    edtV = edtV[pG[0]:-pG[0],pG[1]:-pG[1],:]
                    realMA = realMA[pG[0]:-pG[0],pG[1]:-pG[1],:]

                elif pG[0]==0 and pG[1] > 0 and pG[2] > 0:
                    gridOut = gridOut[:,pG[1]:-pG[1],pG[2]:-pG[2]]
                    realDT = realDT[:,pG[1]:-pG[1],pG[2]:-pG[2]]
                    edtV = edtV[:,pG[1]:-pG[1],pG[2]:-pG[2]]
                    realMA = realMA[:,pG[1]:-pG[1],pG[2]:-pG[2]]

                elif pG[0] > 0 and pG[1] > 0 and pG[2] > 0:
                    gridOut = gridOut[pG[0]:-pG[0],pG[1]:-pG[1],pG[2]:-pG[2]]
                    realDT = realDT[pG[0]:-pG[0],pG[1]:-pG[1],pG[2]:-pG[2]]
                    edtV = edtV[pG[0]:-pG[0],pG[1]:-pG[1],pG[2]:-pG[2]]
                    realMA = realMA[pG[0]:-pG[0],pG[1]:-pG[1],pG[2]:-pG[2]]
                ####################################################################


                ### Reconstruct SubDomains to Check EDT ####
                checkGrid = np.zeros_like(realDT)
                n = 0
                for i in range(0,subDomains[0]):
                    for j in range(0,subDomains[1]):
                        for k in range(0,subDomains[2]):
                            checkGrid[sD[n].indexStart[0]+sD[n].buffer[0][0]: sD[n].indexStart[0]+sD[n].nodes[0]-sD[n].buffer[0][1],
                                      sD[n].indexStart[1]+sD[n].buffer[1][0]: sD[n].indexStart[1]+sD[n].nodes[1]-sD[n].buffer[1][1],
                                      sD[n].indexStart[2]+sD[n].buffer[2][0]: sD[n].indexStart[2]+sD[n].nodes[2]-sD[n].buffer[2][1]] \
                                      = sD[n].grid[sD[n].buffer[0][0] : sD[n].grid.shape[0] - sD[n].buffer[0][1],
                                                      sD[n].buffer[1][0] : sD[n].grid.shape[1] - sD[n].buffer[1][1],
                                                      sD[n].buffer[2][0] : sD[n].grid.shape[2] - sD[n].buffer[2][1]]



                            n = n + 1

                diffGrid = gridOut-checkGrid
                print("L2 Grid Error Norm",np.linalg.norm(diffGrid) )


                checkEDT = np.zeros_like(realDT)
                n = 0
                for i in range(0,subDomains[0]):
                    for j in range(0,subDomains[1]):
                        for k in range(0,subDomains[2]):
                            checkEDT[sD[n].indexStart[0]+sD[n].buffer[0][0]: sD[n].indexStart[0]+sD[n].nodes[0]-sD[n].buffer[0][1],
                                     sD[n].indexStart[1]+sD[n].buffer[1][0]: sD[n].indexStart[1]+sD[n].nodes[1]-sD[n].buffer[1][1],
                                     sD[n].indexStart[2]+sD[n].buffer[2][0]: sD[n].indexStart[2]+sD[n].nodes[2]-sD[n].buffer[2][1]] \
                                     = sDEDT[n].EDT[sD[n].buffer[0][0] : sD[n].grid.shape[0] - sD[n].buffer[0][1],
                                                    sD[n].buffer[1][0] : sD[n].grid.shape[1] - sD[n].buffer[1][1],
                                                    sD[n].buffer[2][0] : sD[n].grid.shape[2] - sD[n].buffer[2][1]]
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
                            checkMA[sD[n].indexStart[0]+sD[n].buffer[0][0]: sD[n].indexStart[0]+sD[n].nodes[0]-sD[n].buffer[0][1],
                                     sD[n].indexStart[1]+sD[n].buffer[1][0]: sD[n].indexStart[1]+sD[n].nodes[1]-sD[n].buffer[1][1],
                                     sD[n].indexStart[2]+sD[n].buffer[2][0]: sD[n].indexStart[2]+sD[n].nodes[2]-sD[n].buffer[2][1]] \
                                     = sDMA[n].MA[sD[n].buffer[0][0] : sD[n].grid.shape[0] - sD[n].buffer[0][1],
                                                    sD[n].buffer[1][0] : sD[n].grid.shape[1] - sD[n].buffer[1][1],
                                                    sD[n].buffer[2][0] : sD[n].grid.shape[2] - sD[n].buffer[2][1]]
                            n = n + 1

                diffMA = np.abs(realMA-checkMA)
                print("L2 MA Error Total Different Voxels",np.sum(diffMA) )



if __name__ == "__main__":
    my_function()
    MPI.Finalize()
