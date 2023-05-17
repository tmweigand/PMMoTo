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


def saveGridData(fileName,rank,Domain,subDomain,**kwargs):

    if rank == 0:
        checkFilePath(fileName)
    comm.barrier()

    allInfo = comm.gather([subDomain.indexStart,subDomain.grid.shape],root=0)

    fileProc = fileName+"/"+fileName.split("/")[-1]+"Proc."
    fileProcLocal = fileName.split("/")[-1]+"/"+fileName.split("/")[-1]+"Proc."
    pointData = {"grid" : subDomain.grid}
    pointDataInfo = {"grid" : (subDomain.grid.dtype, 1)}
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

    allInfo = comm.gather([subDomain.indexStart,subDomain.grid.shape],root=0)

    fileProc = fileName+"/"+fileName.split("/")[-1]+"Proc."
    fileProcLocal = fileName.split("/")[-1]+"/"+fileName.split("/")[-1]+"Proc."
    pointData = {"Phases" : multiPhase.nwpFinal}
    pointDataInfo = {"Phases" : (multiPhase.nwpFinal.dtype, 1)}
         
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

def saveSetData(fileName,rank,Domain,subDomain,setList,**kwargs):

    if rank == 0:
        checkFilePath(fileName)
    comm.barrier()

    ### Place Set Values in Arrays
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

    if rank==0:
        w = vtk.VtkParallelFile(fileName, vtk.VtkPUnstructuredGrid)
        w.openGrid()
        pointData = pointDataInfo
        _addDataToParallelFile(w, cellData=None, pointData=pointData)
        w.openElement("PPoints")
        w.addHeader("points", dtype = x.dtype, ncomp=3)
        w.closeElement("PPoints")

        name = [fileProcLocal]*Domain.numSubDomains
        for nn in range(Domain.numSubDomains):
            name[nn] = name[nn]+str(nn)+".vtu"

        for s in name:
            w.addPiece(start=None,end=None,source=s)
        w.closeGrid()
        w.save()


