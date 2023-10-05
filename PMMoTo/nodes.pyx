# cython: profile=True
# cython: linetrace=True
import math
import numpy as np
cimport numpy as cnp
cimport cython
from mpi4py import MPI
comm = MPI.COMM_WORLD


from . import Orientation
cOrient = Orientation.cOrientation()
cdef int[26][5] directions
cdef int numNeighbors
directions = cOrient.directions
numNeighbors = cOrient.numNeighbors


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def getNodeInfo(rank,grid,phase,inlet,outlet,Domain,loopInfo,subDomain,Orientation):
  """
  Gather information for the nodes. Loop through internal nodes first and
  then go through boundaries.

  Input: Binary grid and Domain,Subdomain,Orientation information,indexOrder

  IndexOrder arranges the ordering of loopInfo so inlet and outlet faces 
  are first to ensure optimal looping and correct boundary values

  Output:
  nodeInfo: [boundary,inlet,outlet,boundaryID,availDirection,lastDirection,visited]
  nodeInfoIndex:[i,j,k,globalIndex,global i,global j,global k]
  nodeDirections: availble directions[26]
  nodeDirectionsIndex: index of availble directions[26]
  """

  numNodes = np.sum(grid)
  nodeInfo = np.zeros([numNodes,7],dtype=np.int8)
  nodeInfo[:,3] = -1 #Initialize BoundaryID
  nodeInfo[:,5] = 25 #Initialize lastDirection
  cdef cnp.int8_t [:,:] _nodeInfo
  _nodeInfo = nodeInfo

  nodeInfoIndex = np.zeros([numNodes,7],dtype=np.uint64)
  cdef cnp.uint64_t [:,:] _nodeInfoIndex
  _nodeInfoIndex = nodeInfoIndex

  nodeDirections = np.zeros([numNodes,26],dtype=np.uint8)
  cdef cnp.uint8_t [:,:] _nodeDirections
  _nodeDirections = nodeDirections

  nodeDirectionsIndex = np.zeros([numNodes,26],dtype=np.uint64)
  cdef cnp.uint64_t [:,:] _nodeDirectionsIndex
  _nodeDirectionsIndex = nodeDirectionsIndex

  nodeTable = -np.ones_like(grid,dtype=np.uint64)
  cdef cnp.uint64_t [:,:,:] _nodeTable
  _nodeTable = nodeTable

  cdef int c,d,i,j,k,ii,jj,kk,availDirection,sInlet,sOutlet
  cdef int iLoc,jLoc,kLoc,globIndex
  cdef int iMin,iMax,jMin,jMax,kMin,kMax
  cdef int _phase = phase

  cdef int numFaces,fIndex
  numFaces = Orientation.numFaces

  cdef int iStart,jStart,kStart
  iStart = subDomain.indexStart[0]
  jStart = subDomain.indexStart[1]
  kStart = subDomain.indexStart[2]

  cdef int iShape,jShape,kShape
  iShape = grid.shape[0]
  jShape = grid.shape[1]
  kShape = grid.shape[2]

  if phase == 0:
    gridP = np.pad(grid,1,constant_values=1)
  else:
    gridP = np.pad(grid,1,constant_values=0)
  cdef cnp.uint8_t [:,:,:] _ind
  _ind = gridP

  cdef cnp.int64_t [:,:,:] _loopInfo
  _loopInfo = loopInfo

  cdef int dN0,dN1,dN2
  dN0 = Domain.nodes[0]
  dN1 = Domain.nodes[1]
  dN2 = Domain.nodes[2]

  # Loop through Boundary Faces to get nodeInfo and nodeIndex
  c = 0
  for fIndex in range(0,numFaces):
    iMin = _loopInfo[fIndex][0][0]
    iMax = _loopInfo[fIndex][0][1]
    jMin = _loopInfo[fIndex][1][0]
    jMax = _loopInfo[fIndex][1][1]
    kMin = _loopInfo[fIndex][2][0]
    kMax = _loopInfo[fIndex][2][1]
    bID = np.asarray(Orientation.faces[fIndex]['ID'],dtype=np.int8)
    sInlet = inlet[fIndex]
    sOutlet = outlet[fIndex]
    for i in range(iMin,iMax):
      for j in range(jMin,jMax):
        for k in range(kMin,kMax):
          if _ind[i+1,j+1,k+1] == _phase:

            iLoc = iStart+i
            jLoc = jStart+j
            kLoc = kStart+k

            if iLoc >= dN0:
              iLoc = 0
            elif iLoc < 0:
              iLoc = dN0-1
            if jLoc >= dN1:
              jLoc = 0
            elif jLoc < 0:
              jLoc = dN1-1
            if kLoc >= dN2:
              kLoc = 0
            elif kLoc < 0:
              kLoc = dN2-1

            globIndex = iLoc*dN1*dN2 +  jLoc*dN2 +  kLoc

            boundaryID = np.copy(bID)
            if (i < 2):
              boundaryID[0] = -1
            elif (i >= iShape-2):
              boundaryID[0] = 1
            if (j < 2):
              boundaryID[1] = -1
            elif (j >= jShape-2):
              boundaryID[1] = 1
            if (k < 2):
              boundaryID[2] = -1
            elif(k >= kShape-2):
              boundaryID[2] = 1

            boundID = cOrient.getBoundaryIDReference(boundaryID)
            _nodeInfo[c,0] = 1
            _nodeInfo[c,1] = sInlet
            _nodeInfo[c,2] = sOutlet
            _nodeInfo[c,3] = boundID
            _nodeInfoIndex[c,0] = i
            _nodeInfoIndex[c,1] = j
            _nodeInfoIndex[c,2] = k
            _nodeInfoIndex[c,3] = globIndex
            _nodeInfoIndex[c,4] = iLoc
            _nodeInfoIndex[c,5] = jLoc
            _nodeInfoIndex[c,6] = kLoc
            _nodeTable[i,j,k] = c
            c = c + 1

  # Loop through internal nodes to get nodeInfo and nodeIndex
  iMin = _loopInfo[numFaces][0][0]
  iMax = _loopInfo[numFaces][0][1]
  jMin = _loopInfo[numFaces][1][0]
  jMax = _loopInfo[numFaces][1][1]
  kMin = _loopInfo[numFaces][2][0]
  kMax = _loopInfo[numFaces][2][1]
  for i in range(iMin,iMax):
    for j in range(jMin,jMax):
      for k in range(kMin,kMax):
        if (_ind[i+1,j+1,k+1] == _phase):
          iLoc = iStart+i
          jLoc = jStart+j
          kLoc = kStart+k
          globIndex = iLoc*dN1*dN2 +  jLoc*dN2 +  kLoc
          _nodeInfoIndex[c,0] = i
          _nodeInfoIndex[c,1] = j
          _nodeInfoIndex[c,2] = k
          _nodeInfoIndex[c,3] = globIndex
          _nodeTable[i,j,k] = c
          c = c + 1

  # Loop through boundary faces to get nodeDirections and _nodeDirectionsIndex
  c = 0
  for fIndex in range(numFaces):
    iMin = _loopInfo[fIndex][0][0]
    iMax = _loopInfo[fIndex][0][1]
    jMin = _loopInfo[fIndex][1][0]
    jMax = _loopInfo[fIndex][1][1]
    kMin = _loopInfo[fIndex][2][0]
    kMax = _loopInfo[fIndex][2][1]
    for i in range(iMin,iMax):
      for j in range(jMin,jMax):
        for k in range(kMin,kMax):
          if _ind[i+1,j+1,k+1] == _phase:
            availDirection = 0
            for d in range(0,numNeighbors):
              ii = directions[d][0]
              jj = directions[d][1]
              kk = directions[d][2]
              if (_ind[i+ii+1,j+jj+1,k+kk+1] == _phase):
                node = _nodeTable[i+ii,j+jj,k+kk]
                _nodeDirections[c,d] = 1
                _nodeDirectionsIndex[c,d] = node
                availDirection += 1

            _nodeInfo[c,4] = availDirection
            c = c + 1

  # Loop through internal nodes to get nodeDirections and _nodeDirectionsIndex
  iMin = _loopInfo[numFaces][0][0]
  iMax = _loopInfo[numFaces][0][1]
  jMin = _loopInfo[numFaces][1][0]
  jMax = _loopInfo[numFaces][1][1]
  kMin = _loopInfo[numFaces][2][0]
  kMax = _loopInfo[numFaces][2][1]
  for i in range(iMin,iMax):
   for j in range(jMin,jMax):
     for k in range(kMin,kMax):
       if _ind[i+1,j+1,k+1] == _phase:
         availDirection = 0
         for d in range(0,numNeighbors):
           ii = directions[d][0]
           jj = directions[d][1]
           kk = directions[d][2]
           if (_ind[i+ii+1,j+jj+1,k+kk+1] == _phase):
             node = _nodeTable[i+ii,j+jj,k+kk]
             _nodeDirections[c,d] = 1
             _nodeDirectionsIndex[c,d] = node
             availDirection += 1

         _nodeInfo[c,4] = availDirection
         c = c + 1

  return [nodeInfo,nodeInfoIndex,nodeDirections,nodeDirectionsIndex]


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def updateMANeighborCount(grid,porousMedia,Orientation,nodeInfo):
  """
  Get Number of Neighbors on Boundary Nodes of Medial Axis with 2 Buffer
  Needed to accurately spoecify type of MA node
  """
  cdef int i,j,k,ii,jj,kk,c,d
  cdef int iMin,iMax,jMin,jMax,kMin,kMax
  cdef int availDirection,node
  cdef int numFaces,fIndex
  numFaces = Orientation.numFaces

  maNodeType = np.copy(nodeInfo[:,4])
  cdef cnp.int8_t [:] _maNodeType
  _maNodeType = maNodeType

  cdef cnp.uint8_t [:,:,:] _ind
  _ind = grid

  cdef cnp.int64_t [:,:,:] loopInfo
  loopInfo = porousMedia.loopInfo


  # Loop through boundary faces to get nodeDirections and _nodeDirectionsIndex
  c = 0
  for fIndex in range(numFaces):
   iMin = loopInfo[fIndex][0][0]
   iMax = loopInfo[fIndex][0][1]
   jMin = loopInfo[fIndex][1][0]
   jMax = loopInfo[fIndex][1][1]
   kMin = loopInfo[fIndex][2][0]
   kMax = loopInfo[fIndex][2][1]
   for i in range(iMin,iMax):
     for j in range(jMin,jMax):
       for k in range(kMin,kMax):
         if _ind[i+1,j+1,k+1] == 1:
           availDirection = 0
           for d in range(0,numNeighbors):
             ii = directions[d][0]
             jj = directions[d][1]
             kk = directions[d][2]
             if (_ind[i+ii+1,j+jj+1,k+kk+1] == 1):
               availDirection += 1

           _maNodeType[c] = availDirection
           c = c + 1

  return maNodeType
