import numpy as np
from mpi4py import MPI
from scipy.ndimage import distance_transform_edt
from edt import edt3d
import time
import pmmoto

def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subDomains = [1,1,1] # Specifies how Domain is broken among procs
    nodes = [300,300,300] # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[0,0],[1,1],[2,2]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[0,0]]
    outlet = [[0,0],[0,0],[0,0]]

    file = './testDomains/1pack.out'

    numSubDomains = np.prod(subDomains)

    testSerial = True
    testAlgo = True


    startTime = time.time()


    domain,sDL,pML = pmmoto.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,"Sphere",file,pmmoto.io.readPorousMediaXYZR)

    edt = pmmoto.filters.calc_edt(sDL,pML.grid)

    ### Save Grid Data where kwargs are used for saving other grid data (i.e. EDT, Medial Axis)
    pmmoto.io.saveGridData("dataOut/grid",rank,domain,sDL,pML.grid,dist=edt)


    if testSerial:

        if rank == 0:
            sD = np.empty((numSubDomains), dtype = object)
            pM = np.empty((numSubDomains), dtype = object)
            sD[0] = sDL
            pM[0] = pML
            for neigh in range(1,numSubDomains):
                sD[neigh] = comm.recv(source=neigh)
                pM[neigh] = comm.recv(source=neigh)

        if rank > 0:
            for neigh in range(1,numSubDomains):
                if rank == neigh:
                    comm.send(sDL,dest=0)
                    comm.send(pML,dest=0)

        if rank==0:
            if testAlgo:
                
                startTime = time.time()

                _,sphereData = pmmoto.io.readPorousMediaXYZR(file)
                # _,sphereData = pmmoto.readPorousMediaLammpsDump(file,rLookupFile)
                
                ##### To GENERATE SINGLE PROC TEST CASE ######
                x = np.linspace(domain.size_domain[0,0]+domain.voxel[0]/2, domain.size_domain[0,1]-domain.voxel[0]/2, nodes[0])
                y = np.linspace(domain.size_domain[1,0]+domain.voxel[1]/2, domain.size_domain[1,1]-domain.voxel[1]/2, nodes[1])
                z = np.linspace(domain.size_domain[2,0]+domain.voxel[2]/2, domain.size_domain[2,1]-domain.voxel[2]/2, nodes[2])
                gridOut = pmmoto.domain_generation.domainGen(x,y,z,sphereData)
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


                realDT = edt3d(gridOut, anisotropy=domain.voxel)
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
                for i in range(0,subDomains[0]):
                    for j in range(0,subDomains[1]):
                        for k in range(0,subDomains[2]):
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
                for i in range(0,subDomains[0]):
                    for j in range(0,subDomains[1]):
                        for k in range(0,subDomains[2]):
                            checkEDT[sD[n].index_start[0]+sD[n].buffer[0]: sD[n].index_start[0]+sD[n].nodes[0]-sD[n].buffer[1],
                                     sD[n].index_start[1]+sD[n].buffer[2]: sD[n].index_start[1]+sD[n].nodes[1]-sD[n].buffer[3],
                                     sD[n].index_start[2]+sD[n].buffer[4]: sD[n].index_start[2]+sD[n].nodes[2]-sD[n].buffer[5]] \
                                     = edt[sD[n].buffer[0] : pM[n].grid.shape[0] - sD[n].buffer[1],
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


                pmmoto.io.saveGridOneProc("dataOut/gridTrueFix",x,y,z,diffEDT)


if __name__ == "__main__":
    my_function()
    MPI.Finalize()
