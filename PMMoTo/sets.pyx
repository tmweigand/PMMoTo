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


### Faces for Faces,Edges,Corners ###
allFaces  = [[ 0,24,20,8],
             [ 1,25,20,8],
             [ 2,20,8],
             [ 3,24,23,8],
             [ 4,25,23,8],
             [ 5,23,8],
             [ 6,24,8],
             [ 7,25,8],
             [ 8],
             [ 9,24,20,17],
             [ 10,25,20,17],
             [ 11,20,17],
             [ 12,24,23,17],
             [ 13,25,23,17],
             [ 14,23,17],
             [ 15,24,17],
             [ 16,25,17],
             [ 17],
             [ 18,24,20],
             [ 19,25,20],
             [ 20],
             [ 21,24,23],
             [ 22,25,23],
             [ 23],
             [ 24],
             [ 25]]


class Set(object):
    def __init__(self, localID = 0, pathID = 0, inlet = False, outlet = False, boundary = False, numNodes = 0, numBoundaryNodes = 0, type = 0, connectedNodes = None):
      self.inlet = inlet
      self.outlet = outlet
      self.boundary = boundary
      self.numBoundaries  = 0
      self.numNodes = numNodes
      self.localID = localID
      self.globalID = 0
      self.pathID = pathID
      self.globalPathID = 0
      self.nodes = np.zeros([numNodes,3],dtype=np.int64)
      self.boundaryNodes = np.zeros(numBoundaryNodes,dtype=np.int64)
      self.boundaryFaces = np.zeros(26,dtype=np.uint8)
      self.boundaryNodeID = np.zeros([numBoundaryNodes,3],dtype=np.int64)
      self.type = type
      self.connectedNodes = connectedNodes
      self.localConnectedSets = []
      self.globalConnectedSets = []
      self.trim = False
      self.minDistance = math.inf
      self.maxDistance = -math.inf
      self.minDistanceNode = -1
      self.maxDistanceNode = -1

    def __lt__(self,obj):
      return ((self.globalID) < (obj.globalID))

    def getNodes(self,n,i,j,k):
      self.nodes[n,0] = i
      self.nodes[n,1] = j
      self.nodes[n,2] = k

    def getAllBoundaryFaces(self,ID):
      faces = allFaces[ID]
      for f in faces:
        self.boundaryFaces[f] = 1
      self.numBoundaries = np.sum(self.boundaryFaces)

    def getBoundaryNodes(self,n,ID,ID2,i,j,k):
      self.boundaryNodes[n] = ID
      self.boundaryFaces[ID2] = 1
      self.getAllBoundaryFaces(ID2)
      self.boundaryNodeID[n,0] = i
      self.boundaryNodeID[n,1] = j
      self.boundaryNodeID[n,2] = k


    def getDistMinMax(self,data):
      for n in self.nodes:
        if data[n[0],n[1],n[2]] < self.minDistance:
          self.minDistance = data[n[0],n[1],n[2]]
          self.minDistanceNode = n
        if data[n[0],n[1],n[2]] > self.maxDistance:
          self.maxDistance = data[n[0],n[1],n[2]]
          self.maxDistanceNode = n




def getBoundarySets(Sets,setCount,subDomain):
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
    for face in range(0,numDirections):
      if bSet.boundaryFaces[face] > 0:

        i = directions[face][0]
        j = directions[face][1]
        k = directions[face][2]

        neighborProc = subDomain.lookUpID[i+nI,j+nJ,k+nK]

        if neighborProc == -1:
          bSet.boundaryFaces[face] = 0
        else:
          if neighborProc not in boundaryData[subDomain.ID]['NeighborProcID'].keys():
            boundaryData[subDomain.ID]['NeighborProcID'][neighborProc] = {'setID':{}}
          bD = boundaryData[subDomain.ID]['NeighborProcID'][neighborProc]
          if bSet.pathID >= 0:
            bD['setID'][bSet.localID] = {'boundaryNodes':bSet.boundaryNodes,
                                         'ProcID':subDomain.ID,
                                         'nodes':bSet.nodes,
                                         'inlet':bSet.inlet,
                                         'outlet':bSet.outlet,
                                         'pathID':bSet.pathID,
                                         'connectedSets':bSet.localConnectedSets}
          else:
            bD['setID'][bSet.localID] = {'boundaryNodes':bSet.boundaryNodes,
                                         'ProcID':subDomain.ID,
                                         'inlet':bSet.inlet,
                                         'outlet':bSet.outlet}

    if (np.sum(bSet.boundaryFaces) == 0):
      boundarySets.remove(bSet)
      bSet.boundary = False

  boundSetCount = len(boundarySets)
  return boundaryData,boundarySets,boundSetCount

def matchProcessorBoundarySets(subDomain,boundaryData,paths):
  """
  Loop through own and neighbor procs and match by boundary nodes
  Input:
  Output: [subDomain.ID,ownSet,nbProc,otherSetKeys[testSetKey],inlet,outlet,ownPath,otherPath]
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
      for otherSet in otherBD[procID]['NeighborProcID'][procID]['setID'].keys():
        countOtherSets += 1
  numSets = np.max([countOwnSets,countOtherSets])
  ####################################################################

  ###########################################################
  ### Loop through own Proc Boundary Data to Find a Match ###
  ###########################################################
  c = 0
  for nbProc in ownBD['NeighborProcID'].keys():
    ownBD_NP =  ownBD['NeighborProcID'][nbProc]
    for ownSet in ownBD_NP['setID'].keys():
      ownBD_Set = ownBD_NP['setID'][ownSet]

      ownNodes  = ownBD_Set['boundaryNodes']
      ownInlet  = ownBD_Set['inlet']
      ownOutlet = ownBD_Set['outlet']
      if paths:
        ownPath = ownBD_Set['pathID']
        ownConnections = ownBD_Set['connectedSets']

      otherBD_NP = otherBD[nbProc]['NeighborProcID'][nbProc]
      otherSetKeys = list(otherBD_NP['setID'].keys())
      numOtherSetKeys = len(otherSetKeys)

      testSetKey = 0
      matchedOut = False
      while testSetKey < numOtherSetKeys:
        inlet = False; outlet = False

        otherNodes  = otherBD_NP['setID'][otherSetKeys[testSetKey]]['boundaryNodes']
        otherInlet  = otherBD_NP['setID'][otherSetKeys[testSetKey]]['inlet']
        otherOutlet = otherBD_NP['setID'][otherSetKeys[testSetKey]]['outlet']
        if paths:
          otherPath = otherBD_NP['setID'][otherSetKeys[testSetKey]]['pathID']
          otherConnections = otherBD_NP['setID'][otherSetKeys[testSetKey]]['connectedSets']

        if len(set(ownNodes).intersection(otherNodes)) > 0:
          if (ownInlet or otherInlet):
            inlet = True
          if (ownOutlet or otherOutlet):
            outlet = True
          if paths:
            matchedSets.append([subDomain.ID,ownSet,nbProc,otherSetKeys[testSetKey],inlet,outlet,ownPath,otherPath])
            matchedSetsConnections.append([subDomain.ID,ownSet,nbProc,otherSetKeys[testSetKey],ownConnections,otherConnections])
            #matchedSets.append([subDomain.ID,ownSet,nbProc,otherSetKeys[testSetKey],inlet,outlet,ownPath,otherPath,ownConnections,otherConnections])
          else:
            matchedSets.append([subDomain.ID,ownSet,nbProc,otherSetKeys[testSetKey],outlet,inlet])
          matchedOut = True
        testSetKey += 1

      if not matchedOut:
        error = True 
        print("Set Not Matched! Hmmm",subDomain.ID,nbProc,ownSet,ownNodes,ownBD_Set['nodes'])


  return matchedSets,matchedSetsConnections,error



def organizePathAndSets(subDomain,size,setData,paths):
  """
  Input: [matchedSets,setCount,boundSetCount,pathCount,boundPathCount]
          from all Procs last two if paths=True
  Matched Sets contains:
    [subDomain.ID,ownSetID,neighProc,neighSetID,Inlet,Outlet]
    OR
    [subDomain.ID,ownSetID,neighProc,neighSetID,Inlet,Outlet,ownPath,otherPath]]

  Output: globalIndexStart,globalBoundarySetID
  """

  if subDomain.ID == 0:

    #########################################
    ### Gather Information from all Procs ###
    #########################################
    allMatchedSets = np.zeros([0,8],dtype=np.int64)
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

    ####################################################
    ### Get Unique Entries and Inlet/Outlet for Sets ###
    ####################################################
    globalSetList = []
    globalInletList = []
    globalOutletList = []
    for s in allMatchedSets:
        if [s[0],s[1]] not in globalSetList:
            globalSetList.append([s[0],s[1]])
            globalInletList.append(s[4])
            globalOutletList.append(s[5])
        else:
            ind = globalSetList.index([s[0],s[1]])
            globalInletList[ind] = s[4]
            globalOutletList[ind] = s[5]
    globalSetID = np.c_[np.asarray(globalSetList),-np.ones(len(globalSetList)),np.asarray(globalInletList),np.asarray(globalOutletList)]
    ####################################################


    ####################################
    ### Loop through assigning setID ###
    ####################################
    cID = 0
    for s in allMatchedSets:
      ind = np.where( (globalSetID[:,0]==s[0]) & (globalSetID[:,1]==s[1]))
      if (globalSetID[ind,2] < 0):
        indNeigh = np.where( (globalSetID[:,0]==s[2]) & (globalSetID[:,1]==s[3]))
        if (globalSetID[indNeigh,2] < 0):
          globalSetID[indNeigh,2] = cID
          globalSetID[ind,2] = cID
          cID += 1
        elif (globalSetID[indNeigh,2] > -1):
          globalSetID[ind,2] = globalSetID[indNeigh,2]
      elif (globalSetID[ind,2] > -1):
        indNeigh = np.where( (globalSetID[:,0]==s[2]) & (globalSetID[:,1]==s[3]))
        if (globalSetID[indNeigh,2] < 0):
          globalSetID[indNeigh,2] = globalSetID[ind,2]
    ####################################


    ###########################################
    ### Prepare Data to send to other procs ###
    ###########################################
    localSetStart = np.zeros(size,dtype=np.int64)
    globalSetScatter = globalSetID
    localSetStart[0] = cID
    for n in range(1,size):
      localSetStart[n] = localSetStart[n-1] + numSets[n-1] - numBoundSets[n-1]
      #globalSetScatter.append(globalSetID[np.where(globalSetID[:,0]==n)])


    #####################################################
    ### Get Unique Entries and Inlet/Outlet for Paths ###
    #####################################################
    globalPathList = []
    globalInletList = []
    globalOutletList = []
    for s in allMatchedSets:
      if [s[0],s[6]] not in globalPathList:
        globalPathList.append([s[0],s[6]])
        globalInletList.append(s[4])
        globalOutletList.append(s[5])
      else:
        ind = globalPathList.index([s[0],s[6]])
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

      if s[6] not in own.keys():
        own[s[6]] = {'Neigh':[[s[2],s[7]]],'Inlet':False,'Outlet':False}
      else:
        if [s[2],s[7]] not in own[s[6]]['Neigh']:
          own[s[6]]['Neigh'].append([s[2],s[7]])

      if s[7] not in other.keys():
        other[s[7]] =  {'Neigh':[[s[0],s[6]]],'Inlet':False,'Outlet':False}
      else:
        if [s[0],s[6]] not in other[s[7]]['Neigh']:
          other[s[7]]['Neigh'].append([s[0],s[6]])

      if not own[s[6]]['Inlet']:
        own[s[6]]['Inlet'] = s[4]
      if not own[s[6]]['Outlet']:
        own[s[6]]['Outlet'] = s[5]
      if not other[s[7]]['Inlet']:
        other[s[7]]['Inlet'] = s[4]
      if not other[s[7]]['Outlet']:
        other[s[7]]['Outlet'] = s[5]
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


def organizeSets(subDomain,size,setData,paths):
  """
  Input: [matchedSets,setCount,boundSetCount,pathCount,boundPathCount]
          from all Procs last two if paths=True
  Matched Sets contains:
    [subDomain.ID,ownSetID,neighProc,neighSetID,Inlet,Outlet]
    OR
    [subDomain.ID,ownSetID,neighProc,neighSetID,Inlet,Outlet,ownPath,otherPath]]

  Output: globalIndexStart,globalBoundarySetID
  """

  if subDomain.ID == 0:

    #############################################
    ### Gather all information from all Procs ###
    #############################################
    allMatchedSets = np.zeros([0,6],dtype=np.int64)
    numSets = np.zeros(size,dtype=np.int64)
    numBoundSets = np.zeros(size,dtype=np.int64)
    for n in range(0,size):
      numSets[n] = setData[n][1]
      numBoundSets[n] = setData[n][2]
      if numBoundSets[n] > 0:
        allMatchedSets = np.append(allMatchedSets,setData[n][0],axis=0)
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

    ####################################################
    ### Get Unique Entries and Inlet/Outlet for Sets ###
    ####################################################
    globalSetList = []
    globalInletList = []
    globalOutletList = []
    for s in allMatchedSets:
        if [s[0],s[1]] not in globalSetList:
            globalSetList.append([s[0],s[1]])
            globalInletList.append(s[4])
            globalOutletList.append(s[5])
        else:
            ind = globalSetList.index([s[0],s[1]])
            globalInletList[ind] = s[4]
            globalOutletList[ind] = s[5]
    globalSetID = np.c_[np.asarray(globalSetList),-np.ones(len(globalSetList)),np.asarray(globalInletList),np.asarray(globalOutletList)]
    ####################################################


    ####################################
    ### Loop through assigning setID ###
    ####################################
    cID = 0
    for s in allMatchedSets:
        ind = np.where( (globalSetID[:,0]==s[0]) & (globalSetID[:,1]==s[1]))
        if (globalSetID[ind,2] < 0):
            indNeigh = np.where( (globalSetID[:,0]==s[2]) & (globalSetID[:,1]==s[3]))
            if (globalSetID[indNeigh,2] < 0):
                globalSetID[indNeigh,2] = cID
                globalSetID[ind,2] = cID
                cID = cID + 1
            elif (globalSetID[indNeigh,2] > -1):
                globalSetID[ind,2] = globalSetID[indNeigh,2]
        elif (globalSetID[ind,2] > -1):
            indNeigh = np.where( (globalSetID[:,0]==s[2]) & (globalSetID[:,1]==s[3]))
            if (globalSetID[indNeigh,2] < 0):
                globalSetID[indNeigh,2] = globalSetID[ind,2]
    ####################################

    ###########################################
    ### Generate Local and Global Numbering ###
    ###########################################
    localSetStart = np.zeros(size,dtype=np.int64)
    globalSetScatter = globalSetID
    localSetStart[0] = cID
    for n in range(1,size):
        localSetStart[n] = localSetStart[n-1] + numSets[n-1] - numBoundSets[n-1]
        #globalSetScatter.append(globalSetID[np.where(globalSetID[:,0]==n)])
    ###########################################

  else:
    localSetStart = None
    globalSetScatter = None

    if paths:
      localPathStart = None
      globalPathScatter = None

  globalIndexStart = comm.scatter(localSetStart, root=0)
  globalBoundarySetID = comm.scatter(globalSetScatter, root=0)

  return globalIndexStart,globalBoundarySetID


def updateSetID(Sets,globalIndexStart,globalBoundarySetID):
  """
  globalBoundarySetID = [subDomain.ID,setLocalID,globalID,Inlet,Outlet]
  """
  c = 0
  for s in Sets:
    if s.boundary == True:
      ind = np.where(globalBoundarySetID[:,1]==s.localID)[0][0]
      s.globalID = int(globalBoundarySetID[ind,2])
      s.inlet = bool(globalBoundarySetID[ind,3])
      s.outlet = bool(globalBoundarySetID[ind,4])
    else:
      s.globalID = globalIndexStart + c
      c = c + 1


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
        globalPathBoundarySetID = np.append(globalPathBoundarySetID,[[rank,s.pathID,newID,s.inlet,s.outlet]],axis=0)
        s.pathID = newID
        c2 = c2 + 1

def setCOMM(Orientation,subDomain,data):
  """
  Transmit data to Neighboring Processors
  """
  dataRecvFace,dataRecvEdge,dataRecvCorner = communication.subDomainComm(Orientation,subDomain,data[subDomain.ID]['NeighborProcID'])

  #############
  ### Faces ###
  #############
  for fIndex in Orientation.faces:
    neigh = subDomain.neighborF[fIndex]
    if (neigh > -1 and neigh != subDomain.ID):
      if neigh in data[subDomain.ID]['NeighborProcID'].keys():
        if neigh not in data:
          data[neigh] = {'NeighborProcID':{}}
        data[neigh]['NeighborProcID'][neigh] = dataRecvFace[fIndex]

  #############
  ### Edges ###
  #############
  for eIndex in Orientation.edges:
    neigh = subDomain.neighborE[eIndex]
    if (neigh > -1 and neigh != subDomain.ID):
      if neigh in data[subDomain.ID]['NeighborProcID'].keys():
        if neigh not in data:
          data[neigh] = {'NeighborProcID':{}}
        data[neigh]['NeighborProcID'][neigh] = dataRecvEdge[eIndex]

  ###############
  ### Corners ###
  ###############
  for cIndex in Orientation.corners:
    neigh = subDomain.neighborC[cIndex]
    if (neigh > -1 and neigh != subDomain.ID):
      if neigh in data[subDomain.ID]['NeighborProcID'].keys():
        if neigh not in data:
          data[neigh] = {'NeighborProcID':{}}
        data[neigh]['NeighborProcID'][neigh] = dataRecvCorner[cIndex]

  return data



def getGlobalConnectedSets(Sets,matchedSets,mathedIDS):
  """
  Update global IDS and use mathedSets to get Gloabl Connections
  """
  #######################################
  ### Update Global Connected Sets ID ###
  #######################################
  for s in Sets:
    if s.localConnectedSets:
      for ss in s.localConnectedSets:
        ID = int(Sets[ss].globalID)
        if ID not in s.globalConnectedSets:
          s.globalConnectedSets.append(ID)
  #######################################


  for s in matchedSets:
    for l in s[5]:
      mID = mathedIDS[s[2]][l]
      if mID not in Sets[s[1]].globalConnectedSets:
        Sets[s[1]].globalConnectedSets.append(mID)
      
