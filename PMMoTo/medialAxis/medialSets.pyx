# cython: profile=True
# cython: linetrace=True
import math
import numpy as np
cimport numpy as cnp
cimport cython
from mpi4py import MPI
comm = MPI.COMM_WORLD
from . import medialSet
from . import medialPath
from .. import communication

cimport cython
from libcpp.vector cimport vector
import numpy as np
from numpy cimport npy_intp, npy_int8, npy_uint8, ndarray, npy_float32
from libcpp cimport bool
from libcpp.map cimport map

cdef struct boundaryData:
    npy_intp ID
    npy_intp procID
    npy_intp pathID
    npy_intp numNodes
    bool inlet
    bool outlet
    bool boundary
    vector[npy_intp] nodes
    vector[npy_intp] boundaryNodes
    vector[npy_intp] connected_sets


cdef struct vertex:
    npy_intp ID
    bool inlet
    bool outlet
    bool boundary
    bool trim
    vector[npy_intp] procID
    vector[npy_intp] connected_sets


from .. import Orientation
cOrient = Orientation.cOrientation()
cdef int[26][5] directions
cdef int numNeighbors
directions = cOrient.directions
numNeighbors = cOrient.numNeighbors

class neighBoundarySets(object):
  def __init__(self,
               setID,
               **kwargs):
    self.setID = setID
    for key, value in kwargs.items():
      setattr(self, key, value)

class medSets(object):
  def __init__(self,
                 sets = None,
                 setCount = 0,
                 pathCount = 0,
                 subDomain = None):
    self.sets = sets
    self.setCount = setCount
    self.pathCount = pathCount 
    self.subDomain = subDomain
    self.boundarySets = []
    self.boundarySetCount = 0
    self.boundaryData = {self.subDomain.ID: {'nProcID':{}}}
    self.nBoundarySets = []
    self.matchedSetData = {self.subDomain.ID: {}}
    self.numConnections = 0
    self.globalMatchedSets = None
    self.localToGlobal = {}
    self.globalToLocal = {}
    self.trimSetData = {self.subDomain.ID: {'setID':{}}}

  def get_boundary_sets(self):
    """
    Get the sets the are on an interal subDomain boundary.
    Check to make sure neighbor is valid procID
    """

    nI = self.subDomain.subID[0] + 1  # PLUS 1 because lookUpID is Padded
    nJ = self.subDomain.subID[1] + 1  # PLUS 1 because lookUpID is Padded
    nK = self.subDomain.subID[2] + 1  # PLUS 1 because lookUpID is Padded

    for set in self.sets:
      procList = []
      if set.boundary:

        for face in range(0,numNeighbors):
          if set.boundaryFaces[face] > 0:
            i = directions[face][0]
            j = directions[face][1]
            k = directions[face][2]

            neighborProc = self.subDomain.lookUpID[i+nI,j+nJ,k+nK]
            if neighborProc < 0:
              set.boundaryFaces[face] = 0
            else:
              procList.append(neighborProc)

        if (np.sum(set.boundaryFaces) == 0):
          set.boundary = False
        else:
          self.boundarySetCount += 1
          set.neighborProcID = procList
          self.boundarySets.append(set)


  def pack_boundary_data(self):
   """
   Collect the Boundary Set Information to Send to neighbor procs
   """
   ID = self.subDomain.ID

   for set in self.boundarySets:
    for nP in set.neighborProcID:
      if nP not in self.boundaryData[ID]['nProcID'].keys():
        self.boundaryData[ID]['nProcID'][nP] = {'setID':{}}
      bD = self.boundaryData[ID]['nProcID'][nP]
      bD['setID'][set.localID] = {'boundaryNodes':set.boundaryNodes,
                                  'numNodes':set.numNodes,
                                  'ProcID':ID,
                                  'nodes':set.nodes,
                                  'inlet':set.inlet,
                                  'outlet':set.outlet,
                                  'pathID':set.pathID,
                                  'connectedSets':set.connectedSets}

  def unpack_boundary_data(self,boundaryData):
    """
    Unpack the boundary data into neighborBoundarySets
    """
    for nProcID in self.boundaryData[self.subDomain.ID]['nProcID'].keys():
      if nProcID == self.subDomain.ID:
        pass
      else:
        for set in boundaryData[nProcID]['nProcID'][nProcID]['setID'].keys():
          nSet = neighBoundarySets(set,**boundaryData[nProcID]['nProcID'][nProcID]['setID'][set])
          self.nBoundarySets.append(nSet)
          

  def match_boundary_sets(self):
    """
    Loop through own boundary and neighbor boundary procs and match by boundary nodes
    """

    numBSets = len(self.nBoundarySets)

    for bSet in self.boundarySets:
      setCount = 0
      match = False
      while setCount < numBSets:
        matchedBNodes = len(set(bSet.boundaryNodes).intersection(self.nBoundarySets[setCount].boundaryNodes))
        if matchedBNodes > 0:
          match = True
          bSet.matchedSet.append(setCount)

          ### Calculate numGlobalNodes for each Set
          if self.subDomain.ID < self.nBoundarySets[setCount].ProcID:
            bSet.numGlobalNodes = bSet.numNodes
          else:
            bSet.numGlobalNodes = bSet.numNodes - matchedBNodes
        
        setCount += 1
      
      if not match:
        print("ERROR Boundary Set Did Not Find a Match. Exiting...")
        communication.raiseError()


  def pack_matched_sets(self):
    """
    Pack the matched boundary set data to globally update 
    """

    self.matchedSetData[self.subDomain.ID] = {'setID':{}}

    for bSet in self.boundarySets:

      inlet = False ; outlet = False
      nProc = []; nSet = []; nPath = []; nConnectedSets = []

      for matchedSet in bSet.matchedSet:
        mSet = self.nBoundarySets[matchedSet]

        if bSet.inlet or mSet.inlet:
          inlet = True
        if bSet.outlet or mSet.outlet:
          outlet = True

        nSet.append(mSet.setID)
        nProc.append(mSet.ProcID)
        nPath.append(mSet.pathID)
        nConnectedSets.append(mSet.connectedSets)

      self.matchedSetData[self.subDomain.ID]['setID'][bSet.localID] = {'nProcID': nProc,
                'nSetID': nSet,
                'inlet': inlet,
                'outlet': outlet, 
                'numGlobalNodes': bSet.numGlobalNodes,
                'pathID': bSet.pathID,
                'nPathID': nPath,
                'connectedSets': bSet.connectedSets,
                'nConnectedSets': nConnectedSets
            }


  def organize_matched_sets(self,allMatchedSetData):
    """
    Propagate matched Set Information - Single Process
    Iterative go through all connected boundary sets
    Grab inlet,outlet,globalID
    """

    if self.subDomain.ID == 0:

      size = len(allMatchedSetData)

      connections = [[] for _ in range(size)]
      ### Go through Data Creating List of Connections
      ### Connections could be from processes that are not neighbors!!
      visited = []
      for n,procData in enumerate(allMatchedSetData):
        for setID in procData[n]['setID'].keys():
          inlet = procData[n]['setID'][setID]['inlet']
          outlet = procData[n]['setID'][setID]['outlet']
          
          queue = []
          setConnect = []
          if (n,setID) not in visited:
            visited.append( (n,setID) )
            queue.append( (n,setID) )
            
            ### Grab all connections
            while queue:
              pID,sID = queue.pop()
              setConnect.append( (pID,sID) )
              nSet = allMatchedSetData[pID][pID]['setID'][sID]
              inlet += nSet['inlet'] 
              outlet += nSet['outlet'] 
              for nProc,cSet in zip(nSet['nProcID'],nSet['nSetID']):
                if (nProc,cSet) not in queue and (nProc,cSet) not in setConnect:
                  queue.append( (nProc,cSet) )
                  visited.append( (nProc,cSet) )

            ### Set values for connections
            if inlet > 0:
              inlet = True
            if outlet > 0:
              outlet = True
            for (nProc,cSet) in setConnect:
              connections[nProc].append({'Sets':(nProc,cSet),'inlet':inlet,'outlet':outlet,'globalID':self.numConnections})
            if setConnect:
              self.numConnections += 1
    else:
      connections = None

    self.globalMatchedSets = comm.scatter(connections, root=0)

  def organize_globalSetID(self,globalIDInfo):
    """
    Generate globalID information for all proccess
    Boundary sets get labeled first, then non-boundary
    """
    if self.subDomain.ID == 0:

      size = len(globalIDInfo)
    
      ### Generate globalID counters
      localSetID = np.zeros(size,dtype=np.int64)
      localSetID[0] = self.numConnections - 1
      for n in range(1,size):
        localSetID[n] = localSetID[n-1] + globalIDInfo[n-1][0] - globalIDInfo[n-1][1]
    else:
      localSetID = None
    self.localSetID = comm.scatter(localSetID, root=0)
        

  def update_globalSetID(self):
    """
    Update inlet,outlet,globalID and inlet/outlet and also save to update connectedSets so using global Indexing
    Note: multiple localSetIDs can map to a single globalID but not the otherway
    """
    for data in self.globalMatchedSets:
      localID = data['Sets'][1]
      self.sets[localID].globalID = data['globalID']
      self.localToGlobal[localID] = data['globalID']

      if data['globalID'] not in self.globalToLocal:
        self.globalToLocal[data['globalID']] = [localID]
        
      if localID not in self.globalToLocal[data['globalID']]:
        self.globalToLocal[data['globalID']].append(localID)

      if data['inlet']:
        self.sets[localID].inlet = True

      if data['outlet']:
        self.sets[localID].outlet = True

    for s in self.sets:
      if not s.boundary:
        s.globalID = self.localSetID
        self.localToGlobal[s.localID] = self.localSetID
        self.globalToLocal[self.localSetID] = [s.localID]
        self.localSetID += 1

  def update_connected_sets(self):
    """
    Create globalConnectedSets
    """
    for s in self.sets:
      for n,cSet in enumerate(s.connectedSets):
        s.globalConnectedSets.append(self.localToGlobal[cSet])

  def collect_paths(self):
    """
    Initialize medialPath and medialPaths
    """
    paths = []
    for nP in range(self.pathCount):
      paths.append( medialPath.medialPath( localID = nP ) )

    for s in self.sets:
      paths[s.pathID].sets.append(s)
      paths[s.pathID].numSets += 1
      if s.inlet:
        paths[s.pathID].inlet = True
      if s.outlet:
        paths[s.pathID].outlet = True
      if s.boundary:
        paths[s.pathID].boundary = True
        paths[s.pathID].boundarySets.append(s)
        paths[s.pathID].boundarySetIDs.append(s.globalID)
        paths[s.pathID].numBoundarySets += 1
    
    medialPaths = medialPath.medialPaths(paths = paths,
                                         pathCount = self.pathCount,
                                         subDomain = self.subDomain)
                            
    return medialPaths

  def trim_sets(self):
    """
    Trim all dead end sets, where dead end is not connected to at least two boundaries.
    The boundaries could be the same (loops).
    First perform a depth first search to grab dead ends. Then trim. 
    """

    visited = []

    for s in self.sets:

      if (s.trim) or (s.localID in visited):
        continue
      else:
        queue = []
        setConnect = []
        visited.append(s.localID)
        queue.append(s.localID)

        while queue:
          cSet = self.sets[queue.pop()]
          setConnect.append(cSet.localID)
          for conSet in cSet.connectedSets:
            if conSet not in visited and conSet not in setConnect:
              queue.append( conSet )
              visited.append( conSet )

        for nSet in reversed(setConnect):
          cSet = self.sets[nSet]
          if cSet.inlet or cSet.outlet or cSet.boundary:
            continue
          elif len(cSet.connectedSets) == 1:
           cSet.trim = True
          else:
            trim = 0
            numCSets = len(cSet.connectedSets)
            for c in cSet.connectedSets:
              if self.sets[c].trim:
                trim += 1
            if trim >= numCSets - 1:
              cSet.trim = True

  def update_trimmed_connected_sets(self):
    """
    Update connectedSets so only non-trimmed values
    """
    for s in self.sets:
      for cSet in s.connectedSets[:]:
        if self.sets[cSet].trim == True:
          s.connectedSets.remove(cSet)
          s.globalConnectedSets.remove(self.localToGlobal[cSet])

  def pack_untrimmed_sets(self):
    """
    Send all untrimmed sets to root to perform a global trim. 
    """
    for s in self.sets:
      if not s.trim:
          self.trimSetData[self.subDomain.ID]['setID'][s.globalID] = {
                'inlet': s.inlet,
                'outlet': s.outlet,
                'boundary':s.boundary,
                'pathID': s.pathID,
                'connectedSets': s.globalConnectedSets,
            }

  def unpack_untrimmed_sets(self,trimSetData):
    """
    Serial Code!
    Organize all untrimmed sets and combine sets on diffrent processes
    Perform a depth first search and trim if not connected to inlet and outlet
    """

    cdef int n,nn,conSet,sID,nP,index
    cdef vector[vertex] setInfo
    cdef vertex node
    cdef map[npy_intp,npy_intp] indexConvert
    cdef bool check

    if self.subDomain.ID == 0:

      n = 0
      for nP,procData in enumerate(trimSetData):
        for sID in procData[nP]['setID'].keys():

          # Not in  Set
          if indexConvert.find(sID) == indexConvert.end():
            node.ID = sID
            indexConvert[sID] = n
            node.procID.push_back( nP )
            node.inlet = procData[nP]['setID'][sID]['inlet']
            node.outlet = procData[nP]['setID'][sID]['outlet']
            node.boundary = procData[nP]['setID'][sID]['boundary']
            node.trim = False

            for conSet in procData[nP]['setID'][sID]['connectedSets']:
              node.connected_sets.push_back(conSet)

            setInfo.push_back(node)

            node.procID.clear()
            node.connected_sets.clear()

            n += 1

          else:
            index = indexConvert[sID]
            check = True
            for nn in range(0,setInfo[index].procID.size()):
              if setInfo[index].procID[nn] == nP:
                check = False
            if check:
              setInfo[index].procID.push_back( nP )


            for conSet in procData[nP]['setID'][sID]['connectedSets']:
              check = True
              for nn in range(0,setInfo[index].connected_sets.size()):
                if setInfo[index].connected_sets[nn] == conSet:
                  check = False
              if check:
                setInfo[index].connected_sets.push_back(conSet)

    return setInfo,indexConvert



  @cython.boundscheck(False)  # Deactivate bounds checking
  @cython.wraparound(False)   # Deactivate negative indexing.
  def serial_trim_sets(self,setInfo,indexMap):
    """
    Serial Code!
    Trim global sets
    """
    
    cdef int n,nn,nnn,sID,conSet,trim,cNode,numCSets

    cdef vector[vertex] vertices
    vertices = setInfo
    cdef int numSet = vertices.size()


    cdef vector[bool] visitedC
    cdef vector[npy_intp] queueC,setConnectC
    cdef map[npy_intp,npy_intp] indexConvert
    indexConvert = indexMap

    for sID in range(0,numSet):
      visitedC.push_back(0)

    if self.subDomain.ID == 0: 

      for n in range(0,numSet):
        if vertices[n].trim == True or visitedC[n] == True:
          continue
        else:

          visitedC[n] = True
          queueC.push_back(n)

          while queueC.size() > 0:
            cNode = queueC.back()
            queueC.pop_back()
            setConnectC.push_back(cNode)

            num_sets = vertices[cNode].connected_sets.size()        
            for nn in range(num_sets):
              conSet = vertices[cNode].connected_sets[nn]
              conSetIndex = indexConvert[conSet]
              if visitedC[conSetIndex] == False:
                queueC.push_back(conSetIndex)
                visitedC[conSetIndex] = True

          for nn in reversed(setConnectC):
            if vertices[nn].inlet or vertices[nn].outlet:
              continue
            elif vertices[nn].connected_sets.size() == 1:
              vertices[nn].trim = True
            else: 
              trim = 0
              numCSets = vertices[nn].connected_sets.size()
              for nnn in range(0,numCSets):
                conSet = vertices[nn].connected_sets[nnn]
                conSetIndex = indexConvert[conSet]
                if vertices[conSetIndex].trim:
                  trim += 1
              if trim >= numCSets - 1:
                vertices[nn].trim = True
                        
          setConnectC.clear()

    return vertices

  def repack_global_trimmed_sets(self,setInfo):
    """
    Serial Code!
    Re-pack setInfo to send out
    """
    if self.subDomain.ID == 0: 
  
      sendSetInfo = [[] for _ in range(self.subDomain.size)]
      for s in setInfo:
        for nP in s['procID']:
           sendSetInfo[nP].append([s['ID'],s['trim']])

    else:
      sendSetInfo = None

    globalTrimData = comm.scatter(sendSetInfo, root=0)

    for s in globalTrimData:
      for localID in self.globalToLocal[s[0]]:
        self.sets[localID].trim = s[1]

    

    