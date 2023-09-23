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
from ..dataOutput import communication
from .. import distance
from .. import morphology
from .. import nodes
from .. import sets
import sys

cdef int numDirections = 26
cdef int[26][5] directions
directions = [[-1,-1,-1,  0, 13],  #0
              [-1,-1, 1,  1, 12],  #1
              [-1,-1, 0,  2, 14],  #2
              [-1, 1,-1,  3, 10],  #3
              [-1, 1, 1,  4,  9],  #4
              [-1, 1, 0,  5, 11],  #5
              [-1, 0,-1,  6, 16],  #6
              [-1, 0, 1,  7, 15],  #7
              [-1, 0, 0,  8, 17],  #8
              [ 1,-1,-1,  9,  4],  #9
              [ 1,-1, 1, 10,  3],  #10
              [ 1,-1, 0, 11,  5],  #11
              [ 1, 1,-1, 12,  1],  #12
              [ 1, 1, 1, 13,  0],  #13
              [ 1, 1, 0, 14,  2],  #14
              [ 1, 0,-1, 15,  7],  #15
              [ 1, 0, 1, 16,  6],  #16
              [ 1, 0, 0, 17,  8],  #17
              [ 0,-1,-1, 18, 22],  #18
              [ 0,-1, 1, 19, 21],  #19
              [ 0,-1, 0, 20, 23],  #20
              [ 0, 1,-1, 21, 19],  #21
              [ 0, 1, 1, 22, 18],  #22
              [ 0, 1, 0, 23, 20],  #23
              [ 0, 0,-1, 24, 25],  #24
              [ 0, 0, 1, 25, 24]]  #25


### Faces/Edges for Faces,Edges,Corners ###
allFaces = [[0, 2, 6, 8, 18, 20, 24],         # 0
            [1, 2, 7, 8, 19, 20, 25],         # 1
            [2, 8, 20],                       # 2
            [3, 5, 6, 8, 21, 23, 24],         # 3
            [4, 5, 7, 8, 22, 23, 25],         # 4
            [5, 8, 23],                       # 5
            [6, 8, 24],                       # 6
            [7, 8, 25],                       # 7
            [8],                              # 8
            [9, 11, 15, 17, 18, 20, 24],      # 9
            [10, 11, 16, 17, 19, 20, 25],     # 10
            [11, 17, 20],                     # 11
            [12, 14, 15, 17, 21, 23, 24],     # 12
            [13, 14, 16, 17, 22, 23, 25],     # 13
            [14, 17, 23],                     # 14
            [15, 17, 24],                     # 15
            [16, 17, 25],                     # 16
            [17],                             # 17
            [18, 20, 24],                     # 18
            [19, 20, 25],                     # 19
            [20],                             # 20
            [21, 23, 24],                     # 21
            [22, 23, 25],                     # 22
            [23],                             # 23
            [24],                             # 24
            [25]]                             # 25


class medialSet(sets.Set):
    def __init__(self, 
                localID = 0, 
                pathID = -1, 
                inlet = False, 
                outlet = False, 
                boundary = False, 
                numNodes = 0, 
                numBoundaryNodes = 0, 
                type = 0, 
                connectedNodes = None):
      self.inlet = inlet
      self.outlet = outlet
      self.boundary = boundary
      self.numBoundaries  = 0
      self.numNodes = numNodes
      self.numBoundaryNodes = numBoundaryNodes
      self.localID = localID
      self.globalID = 0
      self.pathID = pathID
      self.globalPathID = 0
      self.nodes = np.zeros([numNodes,3],dtype=np.uint64) #i,j,k
      self.boundaryNodes = np.zeros(numBoundaryNodes,dtype=np.int64)
      self.boundaryFaces = np.zeros(26,dtype=np.uint8)
      self.boundaryNodeID = np.zeros([numBoundaryNodes,3],dtype=np.int64)
      self.type = type
      self.connectedNodes = connectedNodes
      self.connectedSets = []
      self.globalConnectedSets = []
      self.trim = False
      self.inaccessible = 0
      self.inaccessibleTrim = 0
      self.minDistance = math.inf
      self.maxDistance = -math.inf
      self.minDistanceNode = -1
      self.maxDistanceNode = -1

    def __lt__(self,obj):
      return ((self.globalID) < (obj.globalID))

    def setNodes(self,nodes):
      self.nodes = nodes

    def setBoundaryNodes(self,boundaryNodes,boundaryFaces):
        self.boundaryNodes = boundaryNodes[:,0]
        self.boundaryNodeID = boundaryNodes[:,1:4]
        for bF in boundaryFaces:
            self.getAllBoundaryFaces(bF)

    def getAllBoundaryFaces(self,ID):
      faces = allFaces[ID]
      for f in faces:
        self.boundaryFaces[f] = 1
      self.numBoundaries = np.sum(self.boundaryFaces)

    def getDistMinMax(self,data):
      for n in self.nodes:
        if data[n[0],n[1],n[2]] < self.minDistance:
          self.minDistance = data[n[0],n[1],n[2]]
          self.minDistanceNode = n
        if data[n[0],n[1],n[2]] > self.maxDistance:
          self.maxDistance = data[n[0],n[1],n[2]]
          self.maxDistanceNode = n


def getNodeType(neighbors):
  """
  Determine if Medial Path or Medial Cluster
  """
  pathNode = False
  if neighbors < 3:
    pathNode = True
  return pathNode


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def getSetNodes(set,nNodes,indexMatch,nodeInfo,nodeInfoIndex):
  """
    Add all the nodes and boundaryNodes to the Set class.
    Match the index from MA extraction to nodeIndex
    If interior:
        i,j,k
    If boundaryID add to boundary nodes
        globalIndex,boundaryID,globalI,globalJ,globalK
  """
  cdef cnp.uint64_t [:] _indexMatch
  _indexMatch = indexMatch

  cdef cnp.int8_t [:,:] _nodeInfo
  _nodeInfo = nodeInfo

  cdef cnp.uint64_t [:,:] _nodeInfoIndex
  _nodeInfoIndex = nodeInfoIndex

  cdef int bN,n,ind,cIndex,inNodes,setNodes

  nodes = np.zeros([set.numNodes,3],dtype=np.uint64)
  cdef cnp.uint64_t [:,::1] _nodes
  _nodes = nodes

  bNodes = np.zeros([set.numBoundaryNodes,4],dtype=np.uint64)
  cdef cnp.uint64_t [:,::1] _bNodes
  _bNodes = bNodes

  boundaryFaces = np.zeros(26,dtype=np.uint8)
  cdef cnp.uint8_t [:] _boundaryFaces
  _boundaryFaces = boundaryFaces


  setNodes = set.numNodes
  inNodes = nNodes
  bN =  0
  for n in range(0,setNodes):
    ind = inNodes - setNodes + n
    cIndex = _indexMatch[ind]
    _nodes[n,0] = _nodeInfoIndex[cIndex,0]  #i
    _nodes[n,1] = _nodeInfoIndex[cIndex,1]  #j
    _nodes[n,2] = _nodeInfoIndex[cIndex,2]  #k

    if _nodeInfo[cIndex,0]:
        _bNodes[bN,0] = _nodeInfoIndex[cIndex,3] #globalIndex
        _bNodes[bN,1] = _nodeInfoIndex[cIndex,4] #globalI
        _bNodes[bN,2] = _nodeInfoIndex[cIndex,5] #globalJ
        _bNodes[bN,3] = _nodeInfoIndex[cIndex,6] #globalK
        _boundaryFaces[ _nodeInfo[cIndex,3] ] = 1
        bN = bN + 1

  set.setNodes(nodes)
  set.setBoundaryNodes(bNodes,boundaryFaces)


def getMANodeInfo(cNode,availDirections,numBNodes):
  """
  Get Node Info for Medial Axis
  """

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
def getConnectedMedialAxis(rank,grid,nodeInfo,nodeInfoIndex,nodeDirections,nodeDirectionsIndex,MANodeType):
  """
  Connects the NxNxN  nodes into connected sets.
  1. Path - Exactly 2 Neighbors or 1 Neighbor and on Boundary
  2. Medial Cluster - More than 2 Neighbors

  Create Two Queues for Paths and Clusters

  TO DO: Clean up function call so plassing less variables. Use dictionary?
  """
  cdef int node,ID,nodeValue,d,oppDir,avail,n,index,bN
  cdef int numNodesMA,numSetNodes,numNodes,numBNodes,setCount

  numNodesMA = np.sum(grid)

  indexMatch = np.zeros(numNodesMA,dtype=np.uint64)
  cdef cnp.uint64_t [:] _indexMatch
  _indexMatch = indexMatch

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
              _indexMatch[numNodes] = ID
              _nodeReachDict[ID] = setCount
              pathNode,numBNodes,sBound,sInlet,sOutlet = getMANodeInfo(cNode,MANodeType[ID],numBNodes)
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
            Sets.append(medialSet(localID = setCount,
                               pathID = pathCount,
                               inlet = sInlet,
                               outlet = sOutlet,
                               boundary = sBound,
                               numNodes = numSetNodes,
                               numBoundaryNodes = numBNodes,
                               type = 0,
                               connectedNodes = clusterToPathsConnect))

            getSetNodes(Sets[setCount],numNodes,indexMatch,nodeInfo,nodeInfoIndex)
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
              _indexMatch[numNodes] = ID
              _nodeReachDict[ID] = setCount
              pathNode,numBNodes,sBound,sInlet,sOutlet = getMANodeInfo(cNode,MANodeType[ID],numBNodes)
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
            Sets.append(medialSet(localID = setCount,
                                pathID = pathCount,
                                inlet = sInlet,
                                outlet = sOutlet,
                                boundary = sBound,
                                numNodes = numSetNodes,
                                numBoundaryNodes = numBNodes,
                                type = setType,
                                connectedNodes = pathToClustersConnect))
                                  
            getSetNodes(Sets[setCount],numNodes,indexMatch,nodeInfo,nodeInfoIndex)
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