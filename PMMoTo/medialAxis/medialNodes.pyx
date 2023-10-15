# cython: profile=True
# cython: linetrace=True
import numpy as np
cimport numpy as cnp
cimport cython
from mpi4py import MPI
comm = MPI.COMM_WORLD
from . import medialSet
from . import medialSets

from .. import Orientation
cOrient = Orientation.cOrientation()
cdef int[26][5] directions
cdef int numNeighbors
directions = cOrient.directions
numNeighbors = cOrient.numNeighbors


cdef int getNodeType(int neighbors):
  """
  Determine if Medial Path or Medial Cluster
  """
  pathNode = False
  if neighbors < 3:
    pathNode = True
  return pathNode


cdef getMANodeInfo(cnp.int8_t[:] cNode,int availDirections,int numBNodes,bint sBound,bint sInlet,bint sOutlet):
  """
  Get Node Info for Medial Axis
  """

  cdef int pathNode

  if cNode[0]:  #Boundary
    sBound = True
    numBNodes = numBNodes + 1
    if cNode[1]:  #Inlet
      sInlet = True
    if cNode[2]:  #Outlet
      sOutlet = True

  pathNode = getNodeType(availDirections)

  return pathNode,numBNodes,sBound,sInlet,sOutlet


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def getConnectedMedialAxis(subDomain,grid,Nodes,MANodeType):
  """
  Connects the NxNxN  nodes into connected sets.
  1. Path - Exactly 2 Neighbors or 1 Neighbor and on Boundary
  2. Medial Cluster - More than 2 Neighbors

  Create Two Queues for Paths and Clusters

  """
  cdef int node,ID,nodeValue,d,oppDir,n,proc_ID
  cdef int numNodesMA,numSetNodes = 0,numNodes = 0,numBNodes = 0
  cdef int setCount = 0,pathCount = 0
  cdef bint sInlet,sOutlet,sBound
  cdef list Sets = []

  proc_ID = subDomain.ID

  numNodesMA = np.sum(grid)

  indexMatch = np.zeros(numNodesMA,dtype=np.uint64)
  cdef cnp.uint64_t [:] _indexMatch
  _indexMatch = indexMatch

  nodeReachDict = np.zeros(numNodesMA,dtype=np.uint64)
  cdef cnp.uint64_t [:] _nodeReachDict
  _nodeReachDict = nodeReachDict

  cdef cnp.int8_t [:,:] _nodeInfo
  _nodeInfo = Nodes[0]

  cdef cnp.uint64_t [:,:] _nodeInfoIndex
  _nodeInfoIndex = Nodes[1]

  cdef cnp.uint8_t [:,:] _nodeDirections
  _nodeDirections = Nodes[2]

  cdef cnp.uint64_t [:,:] _nodeDirectionsIndex
  _nodeDirectionsIndex = Nodes[3]

  cdef cnp.int8_t [:] _MANodeType
  _MANodeType = MANodeType

  cdef cnp.int8_t [:] _cNode

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
      _cNode = _nodeInfo[ID]


      # Is Node a Path or Cluster?
      pathNode = getNodeType(_MANodeType[ID])
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
              _cNode = _nodeInfo[ID]
              _indexMatch[numNodes] = ID
              _nodeReachDict[ID] = setCount
              pathNode,numBNodes,sBound,sInlet,sOutlet = getMANodeInfo(_cNode,_MANodeType[ID],numBNodes,sBound,sInlet,sOutlet)
              numSetNodes += 1
              numNodes += 1
              #########################


              ##########################
              ### Find Neighbor Node ###
              ##########################
              while (_cNode[4] > 0):
                nodeValue = -1
                found = False
                d = _cNode[5]
                while d >= 0 and not found:
                  if _nodeDirections[ID,d] == 1:
                    found = True
                    _cNode[4] -= 1
                    _cNode[5] = d
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

              _cNode[6] = 1 #Visited
            ##############################

          ############################
          ### Add Path Set to List ###
          ############################
          if numSetNodes > 0:
            Sets.append(medialSet.medialSet(localID = setCount,
                               proc_ID = proc_ID,
                               pathID = pathCount,
                               inlet = sInlet,
                               outlet = sOutlet,
                               boundary = sBound,
                               numNodes = numSetNodes,
                               numBoundaryNodes = numBNodes,
                               type = 0,
                               connectedNodes = clusterToPathsConnect))

            Sets[setCount].getSetNodes(numNodes,indexMatch,_nodeInfo,_nodeInfoIndex)
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
              _cNode = _nodeInfo[ID]
              _indexMatch[numNodes] = ID
              _nodeReachDict[ID] = setCount
              pathNode,numBNodes,sBound,sInlet,sOutlet = getMANodeInfo(_cNode,_MANodeType[ID],numBNodes,sBound,sInlet,sOutlet)
              numSetNodes += 1
              numNodes += 1
              ########################

              ##########################
              ### Find Neighbor Node ###
              ##########################
              while (_cNode[4] > 0):
                nodeValue = -1
                found = False
                d = _cNode[5]
                while d >= 0 and not found:
                  if _nodeDirections[ID,d] == 1:
                    found = True
                    _cNode[4] -= 1
                    _cNode[5] = d
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

              _cNode[6] = 1 #Visited


          ###############################
          ### Add Cluster Set to List ###
          ###############################
          if numSetNodes > 0:
            setType = 1
            if numSetNodes > 15:
              setType = 2
            Sets.append(medialSet.medialSet(localID = setCount,
                                proc_ID = proc_ID,
                                pathID = pathCount,
                                inlet = sInlet,
                                outlet = sOutlet,
                                boundary = sBound,
                                numNodes = numSetNodes,
                                numBoundaryNodes = numBNodes,
                                type = setType,
                                connectedNodes = pathToClustersConnect))

            Sets[setCount].getSetNodes(numNodes,indexMatch,_nodeInfo,_nodeInfoIndex)
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

  mSets = medialSets.medSets(Sets,setCount,pathCount,subDomain)


  return mSets


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def updateMANeighborCount(grid,porousMedia,Orientation,nodeInfo):
  """
  Get Number of Neighbors on Boundary Nodes of Medial Axis with 2 Buffer
  Needed to accurately spoecify type of MA node
  """
  cdef int i,j,k,ii,jj,kk,c,d
  cdef int iMin,iMax,jMin,jMax,kMin,kMax
  cdef int availDirection
  cdef int numFaces,fIndex
  numFaces = Orientation.numFaces

  maNodeType = np.copy(nodeInfo[:,4])
  cdef cnp.int8_t [:] _maNodeType
  _maNodeType = maNodeType

  cdef cnp.uint8_t [:,:,:] _ind
  _ind = grid

  cdef cnp.int64_t [:,:,:] _loopInfo
  _loopInfo = porousMedia.loopInfo


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
