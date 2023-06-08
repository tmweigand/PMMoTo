import numpy as np
from mpi4py import MPI
from pyevtk.hl import pointsToVTK,gridToVTK, writeParallelVTKGrid,_addDataToParallelFile
from pyevtk import vtk
import os
from . import communication
comm = MPI.COMM_WORLD


def checkFilePath(fileName):

    # Check if Directory exists, if not make it
    paths = fileName.split("/")[0:-1]
    pathWay = ""
    for p in paths:
        pathWay = pathWay+p+"/"
    if not os.path.isdir(pathWay):
        os.makedirs(pathWay)
    
    # Same for individual procs data
    if not os.path.isdir(fileName):
        os.makedirs(fileName)


def saveGridData(fileName,rank,Domain,subDomain,grid,**kwargs):

    if rank == 0:
        checkFilePath(fileName)
    comm.barrier()

    allInfo = comm.gather([subDomain.indexStart,grid.shape],root=0)

    fileProc = fileName+"/"+fileName.split("/")[-1]+"Proc."
    fileProcLocal = fileName.split("/")[-1]+"/"+fileName.split("/")[-1]+"Proc."
    pointData = {"grid" : grid}
    pointDataInfo = {"grid" : (grid.dtype, 1)}
    for key, value in kwargs.items():
        pointData[key]=value
        pointDataInfo[key]= (value.dtype,1)
         
    gridToVTK(fileProc+str(rank), subDomain.x, subDomain.y, subDomain.z,
        start = [subDomain.indexStart[0],subDomain.indexStart[1],subDomain.indexStart[2]],
        pointData = pointData)

    if rank == 0:
        name = [fileProcLocal]*Domain.numSubDomains
        starts = [[0,0,0] for _ in range(Domain.numSubDomains)]
        ends = [[0,0,0] for _ in range(Domain.numSubDomains)]
        nn = 0
        for i in range(0,Domain.subDomains[0]):
            for j in range(0,Domain.subDomains[1]):
                for k in range(0,Domain.subDomains[2]):
                    name[nn] = name[nn]+str(nn)+".vtr"
                    starts[nn][0] = allInfo[nn][0][0]
                    starts[nn][1] = allInfo[nn][0][1]
                    starts[nn][2] = allInfo[nn][0][2]
                    ends[nn][0] = starts[nn][0]+allInfo[nn][1][0]-1
                    ends[nn][1] = starts[nn][1]+allInfo[nn][1][1]-1
                    ends[nn][2] = starts[nn][2]+allInfo[nn][1][2]-1
                    nn = nn + 1

        writeParallelVTKGrid(
            fileName,
            coordsData=((Domain.nodes[0], Domain.nodes[1], Domain.nodes[2]), subDomain.x.dtype),
            starts = starts,
            ends = ends,
            sources = name,
            pointData=pointDataInfo
            )

def saveMultiPhaseData(fileName,rank,Domain,subDomain,multiPhase):

    if rank == 0:
        checkFilePath(fileName)
    comm.barrier()

    allInfo = comm.gather([subDomain.indexStart,multiPhase.mpGrid.shape],root=0)

    fileProc = fileName+"/"+fileName.split("/")[-1]+"Proc."
    fileProcLocal = fileName.split("/")[-1]+"/"+fileName.split("/")[-1]+"Proc."
    pointData = {"Phases" : multiPhase.mpGrid}
    pointDataInfo = {"Phases" : (multiPhase.mpGrid.dtype, 1)}
         
    gridToVTK(fileProc+str(rank), subDomain.x, subDomain.y, subDomain.z,
        start = [subDomain.indexStart[0],subDomain.indexStart[1],subDomain.indexStart[2]],
        pointData = pointData)

    if rank == 0:
        name = [fileProcLocal]*Domain.numSubDomains
        starts = [[0,0,0] for _ in range(Domain.numSubDomains)]
        ends = [[0,0,0] for _ in range(Domain.numSubDomains)]
        nn = 0
        for i in range(0,Domain.subDomains[0]):
            for j in range(0,Domain.subDomains[1]):
                for k in range(0,Domain.subDomains[2]):
                    name[nn] = name[nn]+str(nn)+".vtr"
                    starts[nn][0] = allInfo[nn][0][0]
                    starts[nn][1] = allInfo[nn][0][1]
                    starts[nn][2] = allInfo[nn][0][2]
                    ends[nn][0] = starts[nn][0]+allInfo[nn][1][0]-1
                    ends[nn][1] = starts[nn][1]+allInfo[nn][1][1]-1
                    ends[nn][2] = starts[nn][2]+allInfo[nn][1][2]-1
                    nn = nn + 1

        writeParallelVTKGrid(
            fileName,
            coordsData=((Domain.nodes[0], Domain.nodes[1], Domain.nodes[2]), subDomain.x.dtype),
            starts = starts,
            ends = ends,
            sources = name,
            pointData=pointDataInfo
            )

def saveSetData(fileName,subDomain,setList,**kwargs):

    rank = subDomain.ID
    Domain = subDomain.Domain

    if rank == 0:
        checkFilePath(fileName)
    comm.barrier()

    procSetCounts = comm.allgather(setList.setCount)
    nonZeroProc = np.where(np.asarray(procSetCounts) > 0)[0][0]

    ### Place Set Values in Arrays
    if setList.setCount > 0:
        dim = 0
        for ss in range(0,setList.setCount):
            dim = dim + setList.Sets[ss].numNodes
        x = np.zeros(dim)
        y = np.zeros(dim)
        z = np.zeros(dim)
        setRank = rank*np.ones(dim,dtype=np.uint8)
        globalID = np.zeros(dim,dtype=np.uint64)
        pointData = {"set" : setRank, "globalID" : globalID}
        pointDataInfo = {"set" : (setRank.dtype, 1),
                        "globalID" : (globalID.dtype, 1)
                        }

        ### Handle kwargs
        for key, value in kwargs.items():

            if not hasattr(setList.Sets[0], value):
                if rank == 0:
                    print("Error: Cannot save set data as kwarg %s is not an attribute in Set" %value)
                communication.raiseError()

            dataType = type(getattr(setList.Sets[0],value))
            if dataType == bool: ### pyectk does not support bool?
                dataType = np.uint8
            pointData[key] = np.zeros(dim,dtype=dataType)
            pointDataInfo[key] = (pointData[key].dtype,1)

    
        c = 0
        for ss in range(0,setList.setCount):
            for no in setList.Sets[ss].nodes:
                x[c] = subDomain.x[no[0]]
                y[c] = subDomain.y[no[1]]
                z[c] = subDomain.z[no[2]]

                globalID[c] = setList.Sets[ss].globalID
                for key, value in kwargs.items():
                    pointData[key][c] = getattr(setList.Sets[ss],value)
                c = c + 1

        fileProc = fileName+"/"+fileName.split("/")[-1]+"Proc."
        fileProcLocal = fileName.split("/")[-1]+"/"+fileName.split("/")[-1]+"Proc."
        pointsToVTK(fileProc+str(rank), x,y,z,
            data = pointData 
            )


    if rank == nonZeroProc:
        w = vtk.VtkParallelFile(fileName, vtk.VtkPUnstructuredGrid)
        w.openGrid()
        pointData = pointDataInfo
        _addDataToParallelFile(w, cellData=None, pointData=pointData)
        w.openElement("PPoints")
        w.addHeader("points", dtype = x.dtype, ncomp=3)
        w.closeElement("PPoints")

        procsWithSets = np.count_nonzero(np.array(procSetCounts)>0)
        name = [fileProcLocal]*procsWithSets
        nP = 0
        for nn in range(Domain.numSubDomains):
            if procSetCounts[nn] > 0:
                name[nP] = name[nP]+str(nn)+".vtu"
                nP += 1

        for s in name:
            w.addPiece(start=None,end=None,source=s)
        w.closeGrid()
        w.save()

def saveGrid(fileName,subDomain,grid):

    rank = subDomain.ID
    Domain = subDomain.Domain

    if rank == 0:
        checkFilePath(fileName)
    comm.barrier()

    allInfo = comm.gather([subDomain.indexStart,grid.shape],root=0)

    fileProc = fileName+"/"+fileName.split("/")[-1]+"Proc."
    fileProcLocal = fileName.split("/")[-1]+"/"+fileName.split("/")[-1]+"Proc."
    pointData = {"Grid" : grid}
    pointDataInfo = {"Grid" : (grid.dtype, 1)}
         
    gridToVTK(fileProc+str(rank), subDomain.x, subDomain.y, subDomain.z,
        start = [subDomain.indexStart[0],subDomain.indexStart[1],subDomain.indexStart[2]],
        pointData = pointData)

    if rank == 0:
        name = [fileProcLocal]*Domain.numSubDomains
        starts = [[0,0,0] for _ in range(Domain.numSubDomains)]
        ends = [[0,0,0] for _ in range(Domain.numSubDomains)]
        nn = 0
        for i in range(0,Domain.subDomains[0]):
            for j in range(0,Domain.subDomains[1]):
                for k in range(0,Domain.subDomains[2]):
                    name[nn] = name[nn]+str(nn)+".vtr"
                    starts[nn][0] = allInfo[nn][0][0]
                    starts[nn][1] = allInfo[nn][0][1]
                    starts[nn][2] = allInfo[nn][0][2]
                    ends[nn][0] = starts[nn][0]+allInfo[nn][1][0]-1
                    ends[nn][1] = starts[nn][1]+allInfo[nn][1][1]-1
                    ends[nn][2] = starts[nn][2]+allInfo[nn][1][2]-1
                    nn = nn + 1

        writeParallelVTKGrid(
            fileName,
            coordsData=((Domain.nodes[0], Domain.nodes[1], Domain.nodes[2]), subDomain.x.dtype),
            starts = starts,
            ends = ends,
            sources = name,
            pointData=pointDataInfo
            )



def saveGridOneProc(fileName,x,y,z,grid):

    checkFilePath(fileName)
    pointData = {"Grid" : grid}
         
    gridToVTK(fileName, x, y, z,
        start = [0,0,0],
        pointData = pointData)
    
def saveGridcsv(fileName,subDomain,x,y,z,grid,removeHalo = False):

    rank = subDomain.ID

    if rank == 0:
        checkFilePath(fileName)
    comm.barrier()

    if removeHalo:
        own = subDomain.ownNodesIndex
        size = (own[1]-own[0])*(own[3]-own[2])*(own[5]-own[4])
        printGridOut = np.zeros([size,4])
    else:
        own = np.zeros([6],dtype = np.int64)
        own[1] = grid.shape[0]
        own[3] = grid.shape[1]
        own[4] = grid.shape[2]
        printGridOut = np.zeros([grid.size,4])
                
    fileProc = fileName+"/"+fileName.split("/")[-1]+"Proc."

    c = 0
    for i in range(own[0],own[1]):
        for j in range(own[2],own[3]):
            for k in range(own[4],own[5]):
                printGridOut[c,0] = x[i]
                printGridOut[c,1] = y[j]
                printGridOut[c,2] = z[k]
                printGridOut[c,3] = grid[i,j,k]
                c = c + 1
    
    header = "x,y,z,Grid"
    np.savetxt(fileProc+str(rank)+".csv",printGridOut, delimiter=',',header=header)