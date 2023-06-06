#### One in getConnectedSets
# cython: profile=True
# cython: linetrace=True
import math
import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport malloc, free
import pdb

from mpi4py import MPI
comm = MPI.COMM_WORLD
from . import communication
from . import distance
from . import morphology
import sys

cdef int numDirections = 26
cdef int[26][5] directions
directions =  [[-1,-1,-1,  0, 13],
              [-1,-1, 1,  1, 12],
              [-1,-1, 0,  2, 14],
              [-1, 1,-1,  3, 10],
              [-1, 1, 1,  4,  9],
              [-1, 1, 0,  5, 11],
              [-1, 0,-1,  6, 16],
              [-1, 0, 1,  7, 15],
              [-1, 0, 0,  8, 17],
              [ 1,-1,-1,  9,  4],
              [ 1,-1, 1, 10,  3],
              [ 1,-1, 0, 11,  5],
              [ 1, 1,-1, 12,  1],
              [ 1, 1, 1, 13,  0],
              [ 1, 1, 0, 14,  2],
              [ 1, 0,-1, 15,  7],
              [ 1, 0, 1, 16,  6],
              [ 1, 0, 0, 17,  8],
              [ 0,-1,-1, 18, 22],
              [ 0,-1, 1, 19, 21],
              [ 0,-1, 0, 20, 23],
              [ 0, 1,-1, 21, 19],
              [ 0, 1, 1, 22, 18],
              [ 0, 1, 0, 23, 20],
              [ 0, 0,-1, 24, 25],
              [ 0, 0, 1, 25, 24]]


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int getBoundaryIDReference(cnp.ndarray[cnp.int8_t, ndim=1] boundaryID):
  """
  Method for Determining which type of Boundary and ID
  See direction array for definitions of faces, edges, and corners

  Input: boundaryID[3] corresponding to [x,y,z] and values can be [-1,0,1]
  Output: ID corresponding to face,edge,corner
  """

  cdef int cI,cJ,cK
  cdef int i,j,k
  i = boundaryID[0]
  j = boundaryID[1]
  k = boundaryID[2]

  if i < 0:
    cI = 0
  elif i > 0:
    cI = 9
  else:
    cI = 18

  if j < 0:
    cJ = 0
  elif j > 0:
    cJ = 3
  else:
    cJ = 6

  if k < 0:
    cK = 0
  elif k > 0:
    cK = 1
  else:
    cK = 2

  return cI+cJ+cK

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def getNodeInfo(rank,grid,phase,inlet,outlet,Domain,loopInfo,subDomain,Orientation):
  """
  Gather information for the nodes. Loop through internal nodes first and
  then go through boundaries.

  Input: Binary grid and Domain,Subdomain,Orientation information

  Output:
  nodeInfo: [boundary,inlet,outlet,boundaryID,availDirection,lastDirection,visited]
  nodeInfoIndex:[i,j,k,globalIndex,global i,global j,global k]
  nodeDirections: availble directions[26]
  nodeDirectionsIndex: index of availble directions[26]
  nodeTable: Lookuptable for [i,j,k] = c
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

  nodeTable = -np.ones_like(grid,dtype=np.int64)
  cdef cnp.int64_t [:,:,:] _nodeTable
  _nodeTable = nodeTable

  cdef int c,d,i,j,k,ii,jj,kk,availDirection,perAny,sInlet,sOutlet
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
    perFace  = subDomain.neighborPerF[fIndex]
    perAny = perFace.any()
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

            boundID = getBoundaryIDReference(boundaryID)
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
            for d in range(0,numDirections):
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
         for d in range(0,numDirections):
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

  return nodeInfo,nodeInfoIndex,nodeDirections,nodeDirectionsIndex,nodeTable


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def updateMANeighborCount(grid,subDomain,Orientation,nodeInfo):
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
  loopInfo = subDomain.loopInfo


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
           for d in range(0,numDirections):
             ii = directions[d][0]
             jj = directions[d][1]
             kk = directions[d][2]
             if (_ind[i+ii+1,j+jj+1,k+kk+1] == 1):
               availDirection += 1

           _maNodeType[c] = availDirection
           c = c + 1

  return maNodeType


def getMANodeInfo(cNode,cNodeIndex,maNode,availDirections,numBNodes,setCount,sBound,sInlet,sOutlet):
  """
  Get Node Info for Medial Axis
  """

  maNode[0] = cNodeIndex[0]  #i
  maNode[1] = cNodeIndex[1]  #j
  maNode[2] = cNodeIndex[2]  #k
  if cNode[0]:  #Boundary
    sBound = True
    numBNodes = numBNodes + 1
    maNode[3] = cNode[3]  #BoundaryID
    maNode[4] = cNodeIndex[3] #Global Index
    if cNode[1]:  #Inlet
      sInlet = True
    if cNode[2]:  #Outlet
      sOutlet = True

  maNode[5] = cNodeIndex[4]  #global i
  maNode[6] = cNodeIndex[5]  #global j
  maNode[7] = cNodeIndex[6]  #global k
  maNode[8] = setCount

  pathNode = getNodeType(availDirections)

  return pathNode,numBNodes,sBound,sInlet,sOutlet

def getAllNodeInfo(cNode,cNodeIndex,Node,numBNodes,setCount,sBound,sInlet,sOutlet):
  """
  Get Node Info for Medial Axis
  """

  Node[0] = cNodeIndex[0]  #i
  Node[1] = cNodeIndex[1]  #j
  Node[2] = cNodeIndex[2]  #k
  if cNode[0]:  #Boundary
    sBound = True
    numBNodes = numBNodes + 1
    Node[3] = cNode[3]  #BoundaryID
    Node[4] = cNodeIndex[3] #Global Index
    if cNode[1]:  #Inlet
      sInlet = True
    if cNode[2]:  #Outlet
      sOutlet = True

  Node[5] = cNodeIndex[4]  #global i
  Node[6] = cNodeIndex[5]  #global j
  Node[7] = cNodeIndex[6]  #global k
  Node[8] = setCount

  return numBNodes,sBound,sInlet,sOutlet


def getNodeType(neighbors):
  """
  Determine if Medial Path or Medial Cluster
  """
  pathNode = False
  if neighbors < 3:
    pathNode = True
  return pathNode


def getSetNodes(set,nNodes,_nI):
  cdef int bN,n,ind
  bN =  0
  for n in range(0,set.numNodes):
    ind = nNodes - set.numNodes + n
    set.getNodes(n,_nI[ind,0],_nI[ind,1],_nI[ind,2])
    if _nI[ind,3] > -1:
      set.getBoundaryNodes(bN,_nI[ind,4],_nI[ind,3],_nI[ind,5],_nI[ind,6],_nI[ind,7])
      bN = bN + 1