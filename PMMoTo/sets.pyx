# cython: profile=True
# cython: linetrace=True

import numpy as np
cimport numpy as cnp
cimport cython
from mpi4py import MPI
comm = MPI.COMM_WORLD
from . import communication
from . import nodes

from . import Orientation
cOrient = Orientation.cOrientation()
Orient = Orientation.Orientation()
cdef int[26][5] directions
cdef int numNeighbors
directions = cOrient.directions
numNeighbors = cOrient.numNeighbors

class Set(object):
    def __init__(self, 
                localID = 0, 
                inlet = False, 
                outlet = False, 
                boundary = False, 
                numNodes = 0, 
                numBoundaryNodes = 0):    
      self.localID = localID   
      self.inlet = inlet
      self.outlet = outlet
      self.boundary = boundary
      self.numNodes = numNodes
      self.numBoundaryNodes = numBoundaryNodes
      self.numBoundaries  = 0
      self.numGlobalNodes = numNodes
      self.globalID = 0
      self.nodes = np.zeros([numNodes,3],dtype=np.int64) #i,j,k
      self.boundaryNodes = np.zeros(numBoundaryNodes,dtype=np.int64)
      self.boundaryFaces = np.zeros(26,dtype=np.uint8)
      self.boundaryNodeID = np.zeros([numBoundaryNodes,3],dtype=np.int64)
      self.connectedSets = []
      self.globalConnectedSets = []
      

    def __lt__(self,obj):
      return ((self.globalID) < (obj.globalID))

    def setNodes(self,nodes):
      self.nodes = nodes

    def setBoundaryNodes(self,boundaryNodes,boundaryFaces):
        self.boundary = True
        self.boundaryNodes = boundaryNodes[:,0]
        self.boundaryNodeID = boundaryNodes[:,1:4]
        Orient = Orientation.Orientation()
        allFaces = Orient.allFaces
        for ID,bF in enumerate(boundaryFaces):
          if bF:
            faces = allFaces[ID]
            for f in faces:
              self.boundaryFaces[f] = 1
        self.numBoundaries = np.sum(self.boundaryFaces)


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
                                         'inlet':bSet.inlet,
                                         'outlet':bSet.outlet}

    if (np.sum(bSet.boundaryFaces) == 0):
      boundarySets.remove(bSet)
      bSet.boundary = False

  boundSetCount = len(boundarySets)
  return boundaryData,boundarySets,boundSetCount

def matchProcessorBoundarySets(subDomain,boundaryData):
  """
  Loop through own and neighbor procs and match by boundary nodes
  Input:
  Output: [subDomain.ID,ownSet,nbProc,otherSetKeys[testSetKey],inlet,outlet,otherNodes]
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
          
          matchedSets.append([subDomain.ID,ownSet,nbProc,otherSetKeys[testSetKey],inlet,outlet,setNodes])
          
          matchedOut = True
        testSetKey += 1

      if not matchedOut:
        error = True 
        print("Set Not Matched! Hmmm",subDomain.ID,nbProc,ownSet,ownBNodes)
        #print("Set Not Matched! Hmmm",subDomain.ID,nbProc,ownSet,ownNodes,ownBD_Set['nodes'])


  return matchedSets,matchedSetsConnections,error


def organizeSets(subDomain,size,setData):
  """
  Input: [[subDomain.ID,ownSetID,neighProc,neighSetID,Inlet,Outlet,numSetNodes],
          setCount,
          boundSetCount]
  Output: globalIndexStart,globalBoundarySetID
  """

  if subDomain.ID == 0:

    #############################################
    ### Gather all information from all Procs ###
    #############################################
    allMatchedSets = np.zeros([0,7],dtype=np.int64)
    numSets = np.zeros(size,dtype=np.int64)
    numBoundSets = np.zeros(size,dtype=np.int64)
    for n in range(0,size):
      numSets[n] = setData[n][1]
      numBoundSets[n] = setData[n][2]
      if numBoundSets[n] > 0:
        allMatchedSets = np.append(allMatchedSets,setData[n][0],axis=0)
    allMatchedSets = np.c_[allMatchedSets,-np.ones(allMatchedSets.shape[0],dtype=np.int64)]
    #############################################

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
      if s[7] == -1:
        s[7] = cID
        indexs = np.where( (allMatchedSets[:,0]==s[2])
                        & (allMatchedSets[:,1]==s[3]))[0].tolist()
        while indexs:
          ind = indexs.pop()
          addIndexs  = np.where( (allMatchedSets[:,0]==allMatchedSets[ind,2])
                                & (allMatchedSets[:,1]==allMatchedSets[ind,3])
                                & (allMatchedSets[:,7] == -1) )[0].tolist()
          for aI in addIndexs:
            if aI not in indexs:
              indexs.append(aI)
          allMatchedSets[ind,7] = cID
        cID = cID + 1
    #############################

    ###########################
    ### Sum Total Set Nodes ###
    ###########################
    for n,s in enumerate(allMatchedSets):
      if s[6] > 0:
        indexs = np.where((allMatchedSets[:,0]==s[0]) 
                           & (allMatchedSets[:,1]==s[1]) )[0].tolist()
        while indexs:
          ind = indexs.pop()
          if ind != n:
            allMatchedSets[ind,6] = 0

    for n in range(cID):
      indexs = np.where(allMatchedSets[:,7] == n)[0].tolist() 
      sumNodes = 0
      for nI in indexs:
        sumNodes += allMatchedSets[nI,6]
      for nI in indexs:
        allMatchedSets[nI,6] = sumNodes
    ###########################

    #######################################################
    ### Get Unique Entries and Inlet/Outlet/ID for Sets ###
    #######################################################
    globalSetList = []
    globalInletList = []
    globalOutletList = []
    globalNumNodeList = []
    globalIDList = []
    for s in allMatchedSets:
        if [s[0],s[1]] not in globalSetList:
            globalSetList.append([s[0],s[1]])
            globalInletList.append(s[4])
            globalOutletList.append(s[5])
            globalNumNodeList.append(s[6])
            globalIDList.append(s[7])
        else:
            ind = globalSetList.index([s[0],s[1]])
            globalInletList[ind] = s[4]
            globalOutletList[ind] = s[5]
            globalNumNodeList[ind] = s[6]
            globalIDList[ind] = s[7]

    globalSetID = np.c_[np.asarray(globalSetList,dtype=np.int64),
                        np.asarray(globalIDList,dtype=np.int64),
                        np.asarray(globalInletList,dtype=np.int64),
                        np.asarray(globalOutletList,dtype=np.int64),
                        np.asarray(globalNumNodeList,dtype=np.int64)]
    ####################################################

    ###########################################
    ### Generate Local and Global Numbering ###
    ###########################################
    localSetStart = np.zeros(size,dtype=np.int64)
    globalSetScatter = globalSetID
    localSetStart[0] = cID 
    for n in range(1,size):
        localSetStart[n] = localSetStart[n-1] + numSets[n-1] - numBoundSets[n-1]
    ###########################################

  else:
    localSetStart = None
    globalSetScatter = None

  globalIndexStart = comm.scatter(localSetStart, root=0)
  globalBoundarySetID = comm.bcast(globalSetScatter, root=0)

  return globalIndexStart,globalBoundarySetID


def updateSetID(rank,Sets,globalIndexStart,globalBoundarySetID):
  """
  globalBoundarySetID = [subDomain.ID,setLocalID,globalID,Inlet,Outlet]
  """
  gBSetID = globalBoundarySetID[np.where(globalBoundarySetID[:,0]==rank)]
  c = 0
  for s in Sets:
    if s.boundary == True:
      ind = np.where(gBSetID[:,1]==s.localID)[0][0]
      s.globalID = int(gBSetID[ind,2])
      s.inlet = bool(gBSetID[ind,3])
      s.outlet = bool(gBSetID[ind,4])
      s.numGlobalNodes = int(gBSetID[ind,5])
    else:
      s.globalID = globalIndexStart + c
      c = c + 1

def getGlobalConnectedSets(rank,size,subDomain,Sets,matchedSets,localGlobalIDs,globalLocalIDs):
  """
  Update global IDS and use mathedSets to get Global Connections
  matchedIDs[procID]['localID']=globalID
  """
  #######################################
  ### Update Global Connected Sets ID ###
  #######################################
  for s in Sets:
    if s.connectedSets:
      for ss in s.connectedSets:
        ID = int(Sets[ss].globalID)
        if ID not in s.globalConnectedSets:
          s.globalConnectedSets.append(ID)
  #######################################

  ##############################################################
  ### Loop through Matched Sets to Get Global Connected Sets ###
  ##############################################################
  for n in range(0,size):
    if n in subDomain.lookUpID:
      if n == rank:
        for s in matchedSets[n]:
          for l in s[5]:
            mID = localGlobalIDs[s[2]][l]
            if mID not in Sets[s[1]].globalConnectedSets:
              Sets[s[1]].globalConnectedSets.append(mID)

      else: #Check if Connected Set from Other Proc has More Connections
        for s in matchedSets[n]:
          lID = s[1]
          gID = localGlobalIDs[s[0]][lID]
          if rank in globalLocalIDs[gID].keys():
            
            for l in s[4]:
              mID = localGlobalIDs[s[0]][l]
              for ll in globalLocalIDs[gID][rank]:
                if mID not in Sets[ll].globalConnectedSets:
                  Sets[ll].globalConnectedSets.append(mID)

            for l in s[5]:
              mID = localGlobalIDs[s[2]][l]
              for ll in globalLocalIDs[gID][rank]:
                if mID not in Sets[ll].globalConnectedSets:
                  Sets[ll].globalConnectedSets.append(mID)


def getNodeInfo(cNode,numBNodes,sBound,sInlet,sOutlet):
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

  return numBNodes,sBound,sInlet,sOutlet


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

  nodes = np.zeros([set.numNodes,3],dtype=np.uint64)
  cdef cnp.uint64_t [:,::1] _nodes
  _nodes = nodes

  bNodes = np.zeros([set.numBoundaryNodes,4],dtype=np.uint64)
  cdef cnp.uint64_t [:,::1] _bNodes
  _bNodes = bNodes

  boundaryFaces = np.zeros(26,dtype=np.uint8)
  cdef cnp.uint8_t [:] _boundaryFaces
  _boundaryFaces = boundaryFaces

  cdef int bN,n,ind,cIndex,inNodes,setNodes

  setNodes = set.numNodes
  inNodes = nNodes
  bN = 0
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

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def getConnectedSets(grid,phaseID,Nodes):
  """
  Connects the NxNxN (or NXN) nodes into connected sets.
  1. Inlet
  2. Outlet
  """
  cdef int node,ID,nodeValue,d,oppDir
  cdef int numNodes,numSetNodes,numNodesPhase,setCount

  numNodesPhase = np.count_nonzero(grid==phaseID)

  indexMatch = np.zeros(numNodesPhase,dtype=np.uint64)
  cdef cnp.uint64_t [:] _indexMatch
  _indexMatch = indexMatch

  cdef cnp.int8_t [:,:] _nodeInfo
  _nodeInfo = Nodes[0]

  cdef cnp.uint64_t [:,:] _nodeInfoIndex
  _nodeInfoIndex = Nodes[1]

  cdef cnp.uint8_t [:,:] _nodeDirections
  _nodeDirections = Nodes[2]

  cdef cnp.uint64_t [:,:] _nodeDirectionsIndex
  _nodeDirectionsIndex = Nodes[3]

  cdef cnp.int8_t [:] cNode

  numNodes = 0
  numSetNodes = 0
  numBNodes = 0
  setCount = 0

  Sets = []

  ##############################
  ### Loop Through All Nodes ###
  ##############################
  for node in range(0,numNodesPhase):

    if _nodeInfo[node,6] == 1:  #Visited
      pass
    else:
      ID = node
      cNode = _nodeInfo[ID]
      queue=[node]
      sBound = False; sInlet = False; sOutlet = False

      while queue:

        ########################
        ### Gather Node Info ###
        ########################
        ID = queue.pop(-1)
        if _nodeInfo[ID,6] == 1:
          pass
        else:
          cNode = _nodeInfo[ID]
          _indexMatch[numNodes] = ID
          numBNodes,sBound,sInlet,sOutlet = getNodeInfo(cNode,numBNodes,sBound,sInlet,sOutlet)

          numSetNodes +=  1
          numNodes += 1
        ########################


          ##########################
          ### Find Neighbor Node ###
          ##########################
          while (cNode[4] > 0):
            nodeValue = -1
            found = False
            d = cNode[5]
            while d >= 0  and not found:
              if _nodeDirections[ID,d] == 1:
                found = True
                cNode[4] = cNode[4] - 1
                cNode[5] = d
                oppDir = directions[d][4]
                nodeValue = _nodeDirectionsIndex[ID,d]
                _nodeDirections[nodeValue,oppDir] = 0
                _nodeDirections[ID,d] = 0
              else:
                d = d - 1
            ########################

            #############################
            ### Add Neighbor to Queue ###
            #############################
            if (nodeValue > -1):
              if _nodeInfo[nodeValue,6]:
                pass
              else:
                queue.append(nodeValue)
              _nodeInfo[nodeValue,4] = _nodeInfo[nodeValue,4] - 1
          cNode[6] = 1 #Visited
          ##############################

          ############################
          ### Add Nodes to Set ###
          ############################
      if numSetNodes > 0:
        Sets.append(Set(localID = setCount,
                                  inlet = sInlet,
                                  outlet = sOutlet,
                                  boundary = sBound,
                                  numNodes = numSetNodes,
                                  numBoundaryNodes = numBNodes))

        getSetNodes(Sets[setCount],numNodes,indexMatch,_nodeInfo,_nodeInfoIndex)
        setCount = setCount + 1

      numSetNodes = 0
      numBNodes = 0

  return Sets,setCount

def collectSets(grid,phaseID,inlet,outlet,loopInfo,subDomain):

  rank = subDomain.ID
  size = subDomain.size

  Nodes  = nodes.getNodeInfo(rank,grid,phaseID,inlet,outlet,subDomain.Domain,loopInfo,subDomain,subDomain.Orientation)
  Sets,setCount = getConnectedSets(grid,phaseID,Nodes)

  if size > 1:
    boundaryData,boundarySets,boundSetCount = getBoundarySets(Sets,subDomain)
    boundaryData = communication.setCOMM(subDomain.Orientation,subDomain,boundaryData)
    matchedSets,_,error = matchProcessorBoundarySets(subDomain,boundaryData)
    
    errArray = np.array(error,dtype=np.uint8)
    comm.Allreduce(MPI.IN_PLACE, errArray, op=MPI.MAX)
    if errArray:
        communication.raiseError()

    data = [matchedSets,setCount,boundSetCount]
    data = comm.gather(data, root=0)
    gIS,gBSetID = organizeSets(subDomain,size,data)
    updateSetID(rank,Sets,gIS,gBSetID)

    return Sets,setCount