# cython: profile=True
# cython: linetrace=True
import math
import numpy as np
cimport numpy as cnp
cimport cython
from mpi4py import MPI
comm = MPI.COMM_WORLD
from .. import sets

from .. import Orientation
cOrient = Orientation.cOrientation()
cdef int[26][5] directions
cdef int numNeighbors
directions = cOrient.directions
numNeighbors = cOrient.numNeighbors

class medialSet(sets.Set):
    def __init__(self, 
                 localID = 0, 
                 inlet = False, 
                 outlet = False, 
                 boundary = False, 
                 numNodes = 0, 
                 numBoundaryNodes = 0,
                 pathID = -1, 
                 type = 0, 
                 connectedNodes = None):
      super().__init__(localID, inlet, outlet, boundary, numNodes, numBoundaryNodes)
      self.pathID = pathID
      self.type = type
      self.connectedNodes = connectedNodes
      self.trim = False
      self.inaccessible = 0
      self.inaccessibleTrim = 0
      self.minDistance = math.inf
      self.maxDistance = -math.inf
      self.minDistanceNode = -1
      self.maxDistanceNode = -1

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


def organizePathAndSets(subDomain,size,setData,paths):
  """
  Input: [matchedSets,setCount,boundSetCount,pathCount,boundPathCount]
  Matched Sets contains:
    [subDomain.ID,ownSetID,neighProc,neighSetID,Inlet,Outlet,ownPath,otherPath,globalSetID]]
  Output: globalIndexStart,globalBoundarySetID

  This function collects all of the matched Sets from all Procs. 
  """

  if subDomain.ID == 0:

    #########################################
    ### Gather Information from all Procs ###
    #########################################
    allMatchedSets = np.zeros([0,9],dtype=np.int64)
    numSets = np.zeros(size,dtype=np.int64)
    numBoundSets = np.zeros(size,dtype=np.int64)
    numPaths = np.zeros(size,dtype=np.int64)
    numBoundPaths = np.zeros(size,dtype=np.int64)
    for n in range(0,size):
      numSets[n] = setData[n][1]
      numBoundSets[n] = setData[n][2]
      numPaths[n] = setData[n][3]
      numBoundPaths[n] = setData[n][4]
      if numBoundSets[n] > 0:
        allMatchedSets = np.append(allMatchedSets,setData[n][0],axis=0)
    allMatchedSets = np.c_[allMatchedSets,-np.ones(allMatchedSets.shape[0],dtype=np.int64)]
    #########################################


    ############################
    ### Propagate Inlet Info ###
    ############################
    for s in allMatchedSets:
        if s[4] == 1:
            indexs = np.where( (allMatchedSets[:,0]==s[2])
                             & (allMatchedSets[:,1]==s[3]))[0].tolist()
            while indexs:
                ind = indexs.pop()
                addIndexs  = np.where( (allMatchedSets[:,0]==allMatchedSets[ind,2])
                                     & (allMatchedSets[:,1]==allMatchedSets[ind,3])
                                     & (allMatchedSets[:,4]==0) )[0].tolist()
                if addIndexs:
                    indexs.extend(addIndexs)
                allMatchedSets[ind,4] = 1
    ############################

    #############################
    ### Propagate Outlet Info ###
    #############################
    for s in allMatchedSets:
        if s[5] == 1:
          indexs = np.where( (allMatchedSets[:,0]==s[2])
                           & (allMatchedSets[:,1]==s[3]))[0].tolist()
          while indexs:
            ind = indexs.pop()
            addIndexs  = np.where( (allMatchedSets[:,0]==allMatchedSets[ind,2])
                                 & (allMatchedSets[:,1]==allMatchedSets[ind,3])
                                 & (allMatchedSets[:,5]==0) )[0].tolist()
            if addIndexs:
              indexs.extend(addIndexs)
            allMatchedSets[ind,5] = 1
    #############################

    #############################
    ### Generate globalSetID ###
    #############################
    cID = 0
    for s in allMatchedSets:
      if s[9] == -1:
        s[9] = cID
        indexs = np.where( (allMatchedSets[:,0]==s[2])
                        & (allMatchedSets[:,1]==s[3]))[0].tolist()
        while indexs:
          ind = indexs.pop()
          addIndexs  = np.where( (allMatchedSets[:,0]==allMatchedSets[ind,2])
                                & (allMatchedSets[:,1]==allMatchedSets[ind,3])
                                & (allMatchedSets[:,9] == -1) )[0].tolist()
          for aI in addIndexs:
            if aI not in indexs:
              indexs.append(aI)
          allMatchedSets[ind,9] = cID
        cID = cID + 1
    #############################

    ####################################################
    ### Get Unique Entries and Inlet/Outlet for Sets ###
    ####################################################
    globalSetList = []
    globalInletList = []
    globalOutletList = []
    globalIDList = []
    for s in allMatchedSets:
        if [s[0],s[1]] not in globalSetList:
            globalSetList.append([s[0],s[1]])
            globalInletList.append(s[4])
            globalOutletList.append(s[5])
            globalIDList.append(s[9])
        else:
            ind = globalSetList.index([s[0],s[1]])
            globalInletList[ind] = s[4]
            globalOutletList[ind] = s[5]
            globalIDList[ind] = s[9]
    globalSetID = np.c_[np.asarray(globalSetList),
                        np.asarray(globalIDList,dtype=np.int64),
                        np.asarray(globalInletList),
                        np.asarray(globalOutletList)]
    ####################################################


    ###########################################
    ### Prepare Data to send to other procs ###
    ###########################################
    localSetStart = np.zeros(size,dtype=np.int64)
    globalSetScatter = globalSetID
    localSetStart[0] = cID
    for n in range(1,size):
      localSetStart[n] = localSetStart[n-1] + numSets[n-1] - numBoundSets[n-1]

    #####################################################
    ### Get Unique Entries and Inlet/Outlet for Paths ###
    #####################################################
    globalPathList = []
    globalInletList = []
    globalOutletList = []
    for s in allMatchedSets:
      if [s[0],s[7]] not in globalPathList:
        globalPathList.append([s[0],s[7]])
        globalInletList.append(s[4])
        globalOutletList.append(s[5])
      else:
        ind = globalPathList.index([s[0],s[7]])
        globalInletList[ind] = max(s[4],globalInletList[ind])
        globalOutletList[ind] = max(s[5],globalOutletList[ind])
    globalPathID = np.c_[np.asarray(globalPathList),-np.ones(len(globalPathList)),np.asarray(globalInletList),np.asarray(globalOutletList)]
    #####################################################


    ############################################
    ### Create Dictionary for Sets and Paths ###
    ############################################
    setPathDict = {}
    for c,s in enumerate(allMatchedSets):

      if s[0] not in setPathDict.keys():
        setPathDict[s[0]] = {'Paths':{}}
      if s[2] not in setPathDict.keys():
        setPathDict[s[2]] = {'Paths':{}}

      own = setPathDict[s[0]]['Paths']
      other = setPathDict[s[2]]['Paths']

      if s[7] not in own.keys():
        own[s[7]] = {'Neigh':[[s[2],s[8]]],'Inlet':False,'Outlet':False}
      else:
        if [s[2],s[8]] not in own[s[7]]['Neigh']:
          own[s[7]]['Neigh'].append([s[2],s[8]])

      if s[8] not in other.keys():
        other[s[8]] =  {'Neigh':[[s[0],s[7]]],'Inlet':False,'Outlet':False}
      else:
        if [s[0],s[7]] not in other[s[8]]['Neigh']:
          other[s[8]]['Neigh'].append([s[0],s[7]])

      if not own[s[7]]['Inlet']:
        own[s[7]]['Inlet'] = s[4]
      if not own[s[7]]['Outlet']:
        own[s[7]]['Outlet'] = s[5]
      if not other[s[8]]['Inlet']:
        other[s[8]]['Inlet'] = s[4]
      if not other[s[8]]['Outlet']:
        other[s[8]]['Outlet'] = s[5]
    ############################################


    ##################################################
    ### Loop through All Paths and Gather all Sets ###
    ##################################################
    cID = 0
    for proc in setPathDict.keys():
      for path in setPathDict[proc]['Paths'].keys():
        ind = np.where( (globalPathID[:,0]==proc) & (globalPathID[:,1]==path))
        inlet = globalPathID[ind,3]
        outlet = globalPathID[ind,4]

        if (globalPathID[ind,2] < 0):
          globalPathID[ind,2] = cID

          visited = [[proc,path]]
          queue = setPathDict[proc]['Paths'][path]['Neigh']
          while queue:
            cPP = queue.pop(-1)
            indC =  np.where( (globalPathID[:,0]==cPP[0]) & (globalPathID[:,1]==cPP[1]))

            if (globalPathID[indC,2] < 0):
              globalPathID[indC,2] = cID
            else:
              print("HMMM",globalPathID[indC,2],cID)

            if not inlet:
              inlet = globalPathID[indC,3]
            if not outlet:
              outlet = globalPathID[indC,4]

            for more in setPathDict[cPP[0]]['Paths'][cPP[1]]['Neigh']:
              if more not in queue and more not in visited:
                queue.append(more)
            visited.append(cPP)

          for v in visited:
            indV = np.where( (globalPathID[:,0]==v[0]) & (globalPathID[:,1]==v[1]))
            globalPathID[indV,3] = inlet
            globalPathID[indV,4] = outlet

          cID += 1
    ##################################################


    ###########################################
    ### Generate Local and Global Numbering ###
    ###########################################
    localPathStart = np.zeros(size,dtype=np.int64)
    globalPathScatter = [globalPathID[np.where(globalPathID[:,0]==0)]]
    localPathStart[0] = cID
    for n in range(1,size):
      localPathStart[n] = localPathStart[n-1] + numPaths[n-1] - numBoundPaths[n-1]
      globalPathScatter.append(globalPathID[np.where(globalPathID[:,0]==n)])
    ###########################################

  else:
      localSetStart = None
      globalSetScatter = None
      localPathStart = None
      globalPathScatter = None


  globalIndexStart = comm.scatter(localSetStart, root=0)
  globalBoundarySetID = comm.bcast(globalSetScatter, root=0)
  globalPathIndexStart = comm.scatter(localPathStart, root=0)
  globalPathBoundarySetID = comm.scatter(globalPathScatter, root=0)

  return globalIndexStart,globalBoundarySetID,globalPathIndexStart,globalPathBoundarySetID




def getBoundarySets(Sets,subDomain):
  """
  Get the Sets the are on a valid subDomain Boundary.
  Organize data so sending procID, boundary nodes.
  """

  nI = subDomain.subID[0] + 1  # PLUS 1 because lookUpID is Padded
  nJ = subDomain.subID[1] + 1  # PLUS 1 because lookUpID is Padded
  nK = subDomain.subID[2] + 1  # PLUS 1 because lookUpID is Padded

  boundaryData = {subDomain.ID: {'NeighborProcID':{}}}

  bSetCount = 0
  boundarySets = []

  for set in Sets:
    if set.boundary:
      bSetCount += 1
      boundarySets.append(set)

  for bSet in boundarySets[:]:
    for face in range(0,numNeighbors):
      if bSet.boundaryFaces[face] > 0:

        i = directions[face][0]
        j = directions[face][1]
        k = directions[face][2]

        neighborProc = subDomain.lookUpID[i+nI,j+nJ,k+nK]

        if neighborProc < 0:
          bSet.boundaryFaces[face] = 0
        else:
          if neighborProc not in boundaryData[subDomain.ID]['NeighborProcID'].keys():
            boundaryData[subDomain.ID]['NeighborProcID'][neighborProc] = {'setID':{}}
          bD = boundaryData[subDomain.ID]['NeighborProcID'][neighborProc]
          bD['setID'][bSet.localID] = {'boundaryNodes':bSet.boundaryNodes,
                                         'numNodes':bSet.numNodes,
                                         'ProcID':subDomain.ID,
                                         'nodes':bSet.nodes,
                                         'inlet':bSet.inlet,
                                         'outlet':bSet.outlet,
                                         'pathID':bSet.pathID,
                                         'connectedSets':bSet.connectedSets}

    if (np.sum(bSet.boundaryFaces) == 0):
      boundarySets.remove(bSet)
      bSet.boundary = False

  boundSetCount = len(boundarySets)
  return boundaryData,boundarySets,boundSetCount



def matchProcessorBoundarySets(subDomain,boundaryData):
  """
  Loop through own and neighbor procs and match by boundary nodes
  Input:
  Output: [subDomain.ID,ownSet,nbProc,otherSetKeys[testSetKey],inlet,outlet,otherNodes,ownPath,otherPath]
  """
  otherBD = {}
  matchedSets = []
  matchedSetsConnections = []
  error = False

  ####################################################################
  ### Sort Out Own Proc Bondary Data and Other Procs Boundary Data ###
  ####################################################################
  countOwnSets = 0
  countOtherSets = 0
  for procID in boundaryData.keys():
    if procID == subDomain.ID:
      ownBD = boundaryData[procID]
      for nbProc in ownBD['NeighborProcID'].keys():
        for ownSet in ownBD['NeighborProcID'][nbProc]['setID'].keys():
          countOwnSets += 1

    else:
      otherBD[procID] = boundaryData[procID]
      for _ in otherBD[procID]['NeighborProcID'][procID]['setID'].keys():
        countOtherSets += 1
  ####################################################################

  ###########################################################
  ### Loop through own Proc Boundary Data to Find a Match ###
  ###########################################################
  c = 0
  for nbProc in ownBD['NeighborProcID'].keys():
    ownBD_NP =  ownBD['NeighborProcID'][nbProc]
    for ownSet in ownBD_NP['setID'].keys():
      ownBD_Set = ownBD_NP['setID'][ownSet]

      ownNumNodes = ownBD_Set['numNodes']
      ownBNodes = ownBD_Set['boundaryNodes']
      ownInlet  = ownBD_Set['inlet']
      ownOutlet = ownBD_Set['outlet']
      ownPath = ownBD_Set['pathID']
      ownConnections = ownBD_Set['connectedSets']

      otherBD_NP = otherBD[nbProc]['NeighborProcID'][nbProc]
      otherSetKeys = list(otherBD_NP['setID'].keys())
      numOtherSetKeys = len(otherSetKeys)

      testSetKey = 0
      matchedOut = False
      while testSetKey < numOtherSetKeys:
        inlet = False; outlet = False

        otherNumNodes = otherBD_NP['setID'][otherSetKeys[testSetKey]]['numNodes']
        otherBNodes = otherBD_NP['setID'][otherSetKeys[testSetKey]]['boundaryNodes']
        otherInlet  = otherBD_NP['setID'][otherSetKeys[testSetKey]]['inlet']
        otherOutlet = otherBD_NP['setID'][otherSetKeys[testSetKey]]['outlet']
        otherPath = otherBD_NP['setID'][otherSetKeys[testSetKey]]['pathID']
        otherConnections = otherBD_NP['setID'][otherSetKeys[testSetKey]]['connectedSets']

        matchedBNodes = len(set(ownBNodes).intersection(otherBNodes))
        if matchedBNodes > 0:
          if (ownInlet or otherInlet):
            inlet = True
          if (ownOutlet or otherOutlet):
            outlet = True

          ### Get numGlobalNodes
          if subDomain.ID < nbProc:
            setNodes = ownNumNodes
          else:
            setNodes = ownNumNodes - matchedBNodes
          
          matchedSets.append([subDomain.ID,ownSet,nbProc,otherSetKeys[testSetKey],inlet,outlet,setNodes,ownPath,otherPath])
          matchedSetsConnections.append([subDomain.ID,ownSet,nbProc,otherSetKeys[testSetKey],ownConnections,otherConnections])
          
          matchedOut = True
        testSetKey += 1

      if not matchedOut:
        error = True 
        print("Set Not Matched! Hmmm",subDomain.ID,nbProc,ownSet,ownBNodes)
        #print("Set Not Matched! Hmmm",subDomain.ID,nbProc,ownSet,ownNodes,ownBD_Set['nodes'])


  return matchedSets,matchedSetsConnections,error



def updateSetPathID(rank,Sets,globalIndexStart,globalBoundarySetID,globalPathIndexStart,globalPathBoundarySetID):
  """
  globalBoundarySetID = [subDomain.ID,setLocalID,globalID,Inlet,Outlet]
  globalBoundaryPathID = [subDomain.ID,setLocalID,globalID,Inlet,Outlet]
  """
  gBSetID = globalBoundarySetID[np.where(globalBoundarySetID[:,0]==rank)]
  c = 0; c2 = 0
  for s in Sets:
    if s.boundary == True:
      indS = np.where(gBSetID[:,1]==s.localID)[0][0]
      s.globalID = int(gBSetID[indS,2])
      s.inlet = bool(gBSetID[indS,3])
      s.outlet = bool(gBSetID[indS,4])

      indP = np.where(globalPathBoundarySetID[:,1]==s.pathID)[0][0]
      s.pathID = globalPathBoundarySetID[indP,2]
    else:
      s.globalID = globalIndexStart + c
      c = c + 1
      indP = np.where(globalPathBoundarySetID[:,1]==s.pathID)[0]
      if len(indP)==1:
        indP = indP[0]
        s.pathID = globalPathBoundarySetID[indP,2]
      else:
        newID = globalPathIndexStart + c2
        ## Check if globalPathBoundarySetID exists to correctly initialize 2D append: see issue 20
        if globalPathBoundarySetID.size == 0:
          globalPathBoundarySetID = np.array([[rank,s.pathID,newID,s.inlet,s.outlet]])
        else:
          globalPathBoundarySetID = np.append(globalPathBoundarySetID,[[rank,s.pathID,newID,s.inlet,s.outlet]],axis=0)
        s.pathID = newID
        c2 = c2 + 1