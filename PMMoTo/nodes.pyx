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
from . import sets
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
def getNodeInfo(rank,grid,Domain,subDomain,Orientation):
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

  gridP = np.pad(grid,1,constant_values=0)
  cdef cnp.uint8_t [:,:,:] _ind
  _ind = gridP

  cdef cnp.int64_t [:,:,:] loopInfo
  loopInfo = subDomain.loopInfo

  cdef int dN0,dN1,dN2
  dN0 = Domain.nodes[0]
  dN1 = Domain.nodes[1]
  dN2 = Domain.nodes[2]


  # Loop through Boundary Faces to get nodeInfo and nodeIndex
  c = 0
  for fIndex in range(0,numFaces):
    iMin = loopInfo[fIndex][0][0]
    iMax = loopInfo[fIndex][0][1]
    jMin = loopInfo[fIndex][1][0]
    jMax = loopInfo[fIndex][1][1]
    kMin = loopInfo[fIndex][2][0]
    kMax = loopInfo[fIndex][2][1]
    bID = np.asarray(Orientation.faces[fIndex]['ID'],dtype=np.int8)
    perFace  = subDomain.neighborPerF[fIndex]
    perAny = perFace.any()
    sInlet = subDomain.inlet[fIndex]
    sOutlet = subDomain.outlet[fIndex]
    for i in range(iMin,iMax):
      for j in range(jMin,jMax):
        for k in range(kMin,kMax):
          if _ind[i+1,j+1,k+1] == 1:

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
  iMin = loopInfo[numFaces][0][0]
  iMax = loopInfo[numFaces][0][1]
  jMin = loopInfo[numFaces][1][0]
  jMax = loopInfo[numFaces][1][1]
  kMin = loopInfo[numFaces][2][0]
  kMax = loopInfo[numFaces][2][1]
  for i in range(iMin,iMax):
    for j in range(jMin,jMax):
      for k in range(kMin,kMax):
        if (_ind[i+1,j+1,k+1] == 1):
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
                node = _nodeTable[i+ii,j+jj,k+kk]
                _nodeDirections[c,d] = 1
                _nodeDirectionsIndex[c,d] = node
                availDirection += 1

            _nodeInfo[c,4] = availDirection
            c = c + 1

  # Loop through internal nodes to get nodeDirections and _nodeDirectionsIndex
  iMin = loopInfo[numFaces][0][0]
  iMax = loopInfo[numFaces][0][1]
  jMin = loopInfo[numFaces][1][0]
  jMax = loopInfo[numFaces][1][1]
  kMin = loopInfo[numFaces][2][0]
  kMax = loopInfo[numFaces][2][1]
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
  bN = 0
  for n in range(0,set.numNodes):
    ind = nNodes - set.numNodes + n
    set.getNodes(n,_nI[ind,0],_nI[ind,1],_nI[ind,2])
    if _nI[ind,3] > -1:
      set.getBoundaryNodes(bN,_nI[ind,4],_nI[ind,3],_nI[ind,5],_nI[ind,6],_nI[ind,7])
      bN = bN + 1


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def getConnectedMedialAxis(rank,grid,nodeInfo,nodeInfoIndex,nodeDirections,nodeDirectionsIndex,MANodeType):
  """
  Connects the NxNxN  nodes into connected sets.
  1. Path - Exactly 2 Neighbors or 1 Neighbor and on Boundary
  2. Medial Cluster - More than 2 Neighbors

  Create Two Queues for Paths and Clusters

  TODO: Clean up function call so plassing less variables. Use dictionary?

  """
  cdef int node,ID,nodeValue,d,oppDir,avail,n,index,bN
  cdef int numNodesMA,numSetNodes,numNodes,numBNodes,setCount

  numNodesMA = np.sum(grid)

  nodeIndex = np.zeros([numNodesMA,9],dtype=np.int64)
  cdef cnp.int64_t [:,::1] _nodeIndex
  _nodeIndex = nodeIndex
  for i in range(numNodesMA):
    _nodeIndex[i,3] = -1

  nodeReachDict = np.zeros(numNodesMA,dtype=np.uint64)
  cdef cnp.uint64_t [:] _nodeReachDict
  _nodeReachDict = nodeReachDict

  cdef cnp.int8_t [:,:] _nodeInfo
  _nodeInfo = nodeInfo

  cdef cnp.uint64_t [:,:] _nodeInfoIndex
  _nodeInfoIndex = nodeInfoIndex

  cdef cnp.uint8_t [:,:] _nodeDirections
  _nodeDirections = nodeDirections

  cdef cnp.uint64_t [:,:] _nodeDirectionsIndex
  _nodeDirectionsIndex = nodeDirectionsIndex

  cdef cnp.int8_t [:] cNode
  cdef cnp.uint64_t [:] cNodeIndex
  cdef cnp.int64_t [:] MANodeInfo

  numSetNodes = 0
  numNodes = 0
  numBNodes = 0
  setCount = 0
  pathCount = 0

  Sets = []
  clusterQueues = [] #Store Clusters Identified from Paths
  clusterQueue = []
  pathQueues = []  #Store Paths Identified from Cluster
  pathQueue = []
  clusterToPathsConnect = [] #Track clusters to paths
  pathToClustersConnect = [] #Track paths to clusters

  ##############################
  ### Loop Through All Nodes ###
  ##############################
  for node in range(0,numNodesMA):

    if _nodeInfo[node,6] == 1:  #Visited
      pass
    else:
      ID = node
      cNode = _nodeInfo[ID]


      # Is Node a Path or Cluster?
      pathNode = getNodeType(MANodeType[ID])
      if pathNode:
        pathQueues = [[ID]]
      else:
        clusterQueues = [[ID]]

      #  if Path or Cluster
      while pathQueues or clusterQueues:
        sBound = False; sInlet = False; sOutlet = False

        if pathQueues:
          pathQueue = pathQueues.pop(-1)

          ###############################
          ### Loop through Path Nodes ###
          ###############################
          while pathQueue:

            ########################
            ### Gather Node Info ###
            ########################
            ID = pathQueue.pop(-1)
            if _nodeInfo[ID,6] == 1:
              pass
            else:
              cNode = _nodeInfo[ID]
              cNodeIndex = _nodeInfoIndex[ID,:]
              MANodeInfo = _nodeIndex[numNodes,:]
              _nodeReachDict[ID] = setCount
              pathNode,numBNodes,sBound,sInlet,sOutlet = getMANodeInfo(cNode,cNodeIndex,MANodeInfo,MANodeType[ID],numBNodes,setCount,sBound,sInlet,sOutlet)
              numSetNodes += 1
              numNodes += 1
              #########################


              ##########################
              ### Find Neighbor Node ###
              ##########################
              while (cNode[4] > 0):
                nodeValue = -1
                found = False
                d = cNode[5]
                while d >= 0 and not found:
                  if _nodeDirections[ID,d] == 1:
                    found = True
                    cNode[4] -= 1
                    cNode[5] = d
                    oppDir = directions[d][4]
                    nodeValue = _nodeDirectionsIndex[ID,d]
                    _nodeDirections[nodeValue,oppDir] = 0
                    _nodeDirections[ID,d] = 0
                  else:
                    d -= 1
                ########################

                #############################
                ### Add Neighbor to Queue ###
                #############################
                if (nodeValue > -1):
                  pathNode = getNodeType(MANodeType[nodeValue])
                  if _nodeInfo[nodeValue,6]:
                    pass
                  else:
                    if pathNode:
                      pathQueue.append(nodeValue)
                    else:
                      clusterToPathsConnect.append(nodeValue)
                      clusterQueues.append([nodeValue])
                  _nodeInfo[nodeValue,4] = _nodeInfo[nodeValue,4] - 1

              cNode[6] = 1 #Visited
              ##############################

          ############################
          ### Add Path Set to List ###
          ############################
          if numSetNodes > 0:
            Sets.append(sets.Set(localID = setCount,
                               pathID = pathCount,
                               inlet = sInlet,
                               outlet = sOutlet,
                               boundary = sBound,
                               numNodes = numSetNodes,
                               numBoundaryNodes = numBNodes,
                               type = 0,
                               connectedNodes = clusterToPathsConnect))

            getSetNodes(Sets[setCount],numNodes,_nodeIndex)
            setCount = setCount + 1
          clusterToPathsConnect = []
          numSetNodes = 0
          numBNodes = 0
          ############################


        ##################################
        ### Loop through Cluster Nodes ###
        ##################################
        if clusterQueues:
          clusterQueue = clusterQueues.pop(-1)
          while clusterQueue:

            ########################
            ### Gather Node Info ###
            ########################
            ID = clusterQueue.pop(-1)
            if _nodeInfo[ID,6] == 1:
              pass
            else:
              cNode = _nodeInfo[ID]
              cNodeIndex = _nodeInfoIndex[ID,:]
              MANodeInfo = _nodeIndex[numNodes,:]
              _nodeReachDict[ID] = setCount
              pathNode,numBNodes,sBound,sInlet,sOutlet = getMANodeInfo(cNode,cNodeIndex,MANodeInfo,MANodeType[ID],numBNodes,setCount,sBound,sInlet,sOutlet)
              numSetNodes += 1
              numNodes += 1
              ########################

              ##########################
              ### Find Neighbor Node ###
              ##########################
              while (cNode[4] > 0):
                nodeValue = -1
                found = False
                d = cNode[5]
                while d >= 0 and not found:
                  if _nodeDirections[ID,d] == 1:
                    found = True
                    cNode[4] -= 1
                    cNode[5] = d
                    oppDir = directions[d][4]
                    nodeValue = _nodeDirectionsIndex[ID,d]
                    _nodeDirections[nodeValue,oppDir] = 0
                    _nodeDirections[ID,d] = 0
                  else:
                    d -= 1
              ##########################

                #############################
                ### Add Neighbor to Queue ###
                #############################
                if (nodeValue > -1):
                  pathNode = getNodeType(MANodeType[nodeValue])
                  if _nodeInfo[nodeValue,6]:
                    pass
                  else:
                    if pathNode:
                      pathQueues.append([nodeValue])
                      pathToClustersConnect.append(nodeValue)
                    else:
                      clusterQueue.append(nodeValue)
                  _nodeInfo[nodeValue,4] = _nodeInfo[nodeValue,4] - 1
                #############################

              cNode[6] = 1 #Visited


          ###############################
          ### Add Cluster Set to List ###
          ###############################
          if numSetNodes > 0:
            setType = 1
            if numSetNodes > 15:
              setType = 2
            Sets.append(sets.Set(localID = setCount,
                                pathID = pathCount,
                                inlet = sInlet,
                                outlet = sOutlet,
                                boundary = sBound,
                                numNodes = numSetNodes,
                                numBoundaryNodes = numBNodes,
                                type = setType,
                                connectedNodes = pathToClustersConnect))
                                  
            getSetNodes(Sets[setCount],numNodes,_nodeIndex)
            setCount = setCount + 1
          pathToClustersConnect = []
          numSetNodes = 0
          numBNodes = 0
          ###############################

      pathCount += 1


  ###########################
  ### Grab Connected Sets ###
  ###########################
  for s in Sets:
    for n in s.connectedNodes:
      ID = _nodeReachDict[n]
      if ID not in s.connectedSets:
        s.connectedSets.append(ID)
      if ID not in Sets[ID].connectedSets:
        Sets[ID].connectedSets.append(s.localID)
  ###########################


  return Sets,setCount,pathCount
