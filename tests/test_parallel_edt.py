import numpy as np
from mpi4py import MPI
from scipy.ndimage import distance_transform_edt
import edt
import time
import pmmoto
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import matplotlib.pyplot as plt


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

    subdomains = [3,3,3] # Specifies how Domain is broken among rrocs
    nodes = [300,300,300] # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[2,2],[2,2],[2,2]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[0,0]]
    outlet = [[0,0],[0,0],[0,0]]

    file = './testDomains/50pack.out'

    numSubDomains = np.prod(subdomains)


    testSerial = True
    testAlgo = True


    startTime = time.time()


    domain,sDL,pML = pmmoto.genDomainSubDomain(rank,size,subdomains,nodes,boundaries,inlet,outlet,"Sphere",file,pmmoto.readPorousMediaXYZR)

    sD_EDT = pmmoto.calcEDT(sDL,pML.grid,stats = True,sendClass=True)

    endTime = time.time()
    print("PMMoTo run time:",endTime-startTime)

    ### Save Grid Data where kwargs are used for saving other grid data (i.e. EDT, Medial Axis)
    pmmoto.saveGridData("dataOut/grid_correct",rank,domain,sDL,pML.grid,dist=sD_EDT.EDT)


    if testSerial:

        if rank == 0:
            sD = np.empty((numSubDomains), dtype = object)
            sDEDT = np.empty((numSubDomains), dtype = object)
            pM = np.empty((numSubDomains), dtype = object)
            sD[0] = sDL
            sDEDT[0] = sD_EDT
            pM[0] = pML
            for neigh in range(1,numSubDomains):
                sD[neigh] = comm.recv(source=neigh)
                sDEDT[neigh] = comm.recv(source=neigh)
                pM[neigh] = comm.recv(source=neigh)

        if rank > 0:
            for neigh in range(1,numSubDomains):
                if rank == neigh:
                    comm.send(sDL,dest=0)
                    comm.send(sD_EDT,dest=0)
                    comm.send(pML,dest=0)


        if rank==0:
            if testAlgo:
                
                startTime = time.time()

                _,sphereData = PMMoTo.readPorousMediaXYZR(file)
                # _,sphereData = PMMoTo.readPorousMediaLammpsDump(file,rLookupFile)
                
                ##### To GENERATE SINGLE PROC TEST CASE ######
                x = np.linspace(domain.size_domain[0,0]+domain.voxel[0]/2, domain.size_domain[0,1]-domain.voxel[0]/2, nodes[0])
                y = np.linspace(domain.size_domain[1,0]+domain.voxel[1]/2, domain.size_domain[1,1]-domain.voxel[1]/2, nodes[1])
                z = np.linspace(domain.size_domain[2,0]+domain.voxel[2]/2, domain.size_domain[2,1]-domain.voxel[2]/2, nodes[2])
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


                realDT = edt.edt3d(gridOut, anisotropy=domain.voxel)
                edtV,_ = distance_transform_edt(gridOut,sampling=domain.voxel,return_indices=True)
                endTime = time.time()

                print("Serial Time:",endTime-startTime)

                arrayList = [gridOut,realDT,edtV]

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

                ### Reconstruct SubDomains to Check EDT ####
                checkGrid = np.zeros_like(realDT)
                n = 0
                for i in range(0,subdomains[0]):
                    for j in range(0,subdomains[1]):
                        for k in range(0,subdomains[2]):
                            checkGrid[sD[n].index_start[0]+sD[n].buffer[0]: sD[n].index_start[0]+sD[n].nodes[0]-sD[n].buffer[1],
                                      sD[n].index_start[1]+sD[n].buffer[2]: sD[n].index_start[1]+sD[n].nodes[1]-sD[n].buffer[3],
                                      sD[n].index_start[2]+sD[n].buffer[4]: sD[n].index_start[2]+sD[n].nodes[2]-sD[n].buffer[5]] \
                                      = pM[n].grid[sD[n].buffer[0] : pM[n].grid.shape[0] - sD[n].buffer[1],
                                                      sD[n].buffer[2] : pM[n].grid.shape[1] - sD[n].buffer[3],
                                                      sD[n].buffer[4] : pM[n].grid.shape[2] - sD[n].buffer[5]]



                            n = n + 1

                diffGrid = gridOut-checkGrid
                print("L2 Grid Error Norm",np.linalg.norm(diffGrid) )


                checkEDT = np.zeros_like(realDT)
                n = 0
                for i in range(0,subdomains[0]):
                    for j in range(0,subdomains[1]):
                        for k in range(0,subdomains[2]):
                            checkEDT[sD[n].index_start[0]+sD[n].buffer[0]: sD[n].index_start[0]+sD[n].nodes[0]-sD[n].buffer[1],
                                     sD[n].index_start[1]+sD[n].buffer[2]: sD[n].index_start[1]+sD[n].nodes[1]-sD[n].buffer[3],
                                     sD[n].index_start[2]+sD[n].buffer[4]: sD[n].index_start[2]+sD[n].nodes[2]-sD[n].buffer[5]] \
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

                PMMoTo.saveGridOneProc("dataOut/gridTrue",x,y,z,diffEDT)


if __name__ == "__main__":
    my_function()
    MPI.Finalize()
