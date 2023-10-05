# cython: profile=True
# cython: linetrace=True
import math
import numpy as np
cimport numpy as cnp
cimport cython
from .. import communication
from mpi4py import MPI
comm = MPI.COMM_WORLD

class neighBoundaryPaths(object):
  def __init__(self,
               pathID,
               **kwargs):
    self.pathID = pathID
    for key, value in kwargs.items():
      setattr(self, key, value)

class medialPath(object):
    def __init__(self, localID = 0):
        self.localID = localID
        self.sets = []
        self.inlet = False
        self.outlet = False
        self.boundary = False
        self.numSets = 0
        self.numBoundarySets = 0
        self.boundarySets = []
        self.boundarySetIDs = []
        self.boundaryProcAndSet = {'nProcID':{}}
        self.matchedPath = []
        self.nBoundaryPaths = []


class medialPaths(object):
    def __init__(self,
                 paths = [],
                 pathCount = 0,
                 subDomain = None):
        self.paths = paths
        self.pathCount = pathCount
        self.subDomain = subDomain
        self.boundaryPaths = []
        self.boundaryPathCount = 0
        self.boundaryData = {self.subDomain.ID: {'nProcID':{}}}
        self.nBoundaryPaths = []
        self.matchedPathData = {self.subDomain.ID: {}}
        self.numConnections = 0
        self.localPathID = 0
        self.localToGlobal = {}
    
    def get_boundary_paths(self):
        """
        Get the paths the are on an interal (and periodic) subDomain boundary.
        Already check sets to make sure neighbor is valid procID
        """
        for p in self.paths:
            if p.boundary:
                self.boundaryPaths.append(p)
                self.boundaryPathCount += 1
                for bSet in p.boundarySets:
                    for nP in bSet.neighborProcID:
                        if nP not in p.boundaryProcAndSet['nProcID'].keys():
                            p.boundaryProcAndSet['nProcID'][nP] = []
                        p.boundaryProcAndSet['nProcID'][nP].append(bSet.globalID)

    def pack_boundary_data(self):
        """
        Collect the Boundary Path Information to Send to Neighbor Procs
        """
        ID = self.subDomain.ID

        for path in self.boundaryPaths:
            for nP in path.boundaryProcAndSet['nProcID'].keys():
                if nP not in self.boundaryData[ID]['nProcID'].keys():
                    self.boundaryData[ID]['nProcID'][nP] = {'pathID':{}}
                bD = self.boundaryData[ID]['nProcID'][nP]
                bD['pathID'][path.localID] = {'boundarySets':path.boundaryProcAndSet['nProcID'][nP],
                                                  'ProcID':ID,
                                                  'inlet':path.inlet,
                                                  'outlet':path.outlet}

    def unpack_boundary_data(self,boundaryData):
        """
        Unpack the boundary data into neighborBoundaryPaths
        """
        for nProcID in self.boundaryData[self.subDomain.ID]['nProcID'].keys():
            if nProcID == self.subDomain.ID:
                pass
            else:
                for path in boundaryData[nProcID]['nProcID'][nProcID]['pathID'].keys():
                    nPath = neighBoundaryPaths(path,**boundaryData[nProcID]['nProcID'][nProcID]['pathID'][path])
                    self.nBoundaryPaths.append(nPath)

    def match_boundary_paths(self):
        """
        Loop through own boundary and neighbor boundary procs and match by boundary nodes
        """

        numBPaths = len(self.nBoundaryPaths)

        for bPath in self.boundaryPaths:
            pathCount = 0
            match = False
            while pathCount < numBPaths:
                matchedBSets = len(set(bPath.boundarySetIDs).intersection(self.nBoundaryPaths[pathCount].boundarySets))
                if matchedBSets > 0:
                    match = True
                    bPath.matchedPath.append(pathCount)
        
                pathCount += 1
      
            if not match:
                print("ERROR Boundary Set Did Not Find a Match. Exiting...")
                communication.raiseError()

    def pack_matched_paths(self):
        """
        Pack the matched boundary path data to globally update 
        """

        self.matchedPathData[self.subDomain.ID] = {'pathID':{}}

        for bPath in self.boundaryPaths:

            inlet = False; outlet = False
            nPathID = [];

            for matchedPath in bPath.matchedPath:
                mPath = self.nBoundaryPaths[matchedPath]

                if bPath.inlet or mPath.inlet:
                  inlet = True
                if bPath.outlet or mPath.outlet:
                  outlet = True

                if (mPath.ProcID,mPath.pathID) not in nPathID:
                  nPathID.append( (mPath.ProcID,mPath.pathID) )

            self.matchedPathData[self.subDomain.ID]['pathID'][bPath.localID] = {'nPathID': nPathID,
                'inlet': inlet,
                'outlet': outlet, 
                }

    def organize_matched_paths(self,allMatchedPathData):
        """
        Propagate matched Set Information - Single Process
        Iterative go through all connected boundary sets
        Grab inlet,outlet,globalID
        """

        if self.subDomain.ID == 0:

            size = len(allMatchedPathData)

            connections = [[] for _ in range(size)]
            ### Go through Data Creating List of Connections
            ### Connections could be from processes that are not neighbors!!
            visited = []
            for n,procData in enumerate(allMatchedPathData):
                for pathID in procData[n]['pathID'].keys():
                    inlet = procData[n]['pathID'][pathID]['inlet']
                    outlet = procData[n]['pathID'][pathID]['outlet']
                
                    queue = []
                    pathConnect = []
                    if (n,pathID) not in visited:
                        visited.append( (n,pathID) )
                        queue.append( (n,pathID) )
                    
                    ### Grab all connections
                    while queue:
                        pID,sID = queue.pop()
                        pathConnect.append( (pID,sID) )
                        nPath = allMatchedPathData[pID][pID]['pathID'][sID]
                        inlet += nPath['inlet'] 
                        outlet += nPath['outlet'] 
                        for nPathID in nPath['nPathID']:
                          if nPathID not in queue and nPathID not in pathConnect:
                            queue.append( nPathID )
                            visited.append( nPathID )

                    ### Set values for connections
                    if inlet > 0:
                        inlet = True
                    if outlet > 0:
                        outlet = True
                    for (nProc,cPath) in pathConnect:
                        connections[nProc].append({'Paths':(nProc,cPath),'inlet':inlet,'outlet':outlet,'globalID':self.numConnections})
                    if pathConnect:
                      self.numConnections += 1
        else:
          connections = None

        self.globalMatchedPaths = comm.scatter(connections, root=0)

    def organize_globalPathID(self,globalIDInfo):
      """
      Generate globalID information for all proccess
      Boundary paths get labeled first, then non-boundary
      """
      if self.subDomain.ID == 0:

        size = len(globalIDInfo)
      
        ### Generate globalID counters
        localPathID = np.zeros(size,dtype=np.int64)
        localPathID[0] = self.numConnections - 1
        for n in range(1,size):
          localPathID[n] = localPathID[n-1] + globalIDInfo[n-1][0] - globalIDInfo[n-1][1]
      else:
        localPathID = None
      self.localPathID = comm.scatter(localPathID, root=0)

    def update_globalPathID(self):
      """
      Update inlet,outlet,globalID and also save to update connectedSets so using global Indexing
      """

      for data in self.globalMatchedPaths:
        localID = data['Paths'][1]
        self.paths[localID].globalID = data['globalID']
        self.localToGlobal[localID] = data['globalID']
        if data['inlet']:
          self.paths[localID].inlet = True
        if data['outlet']:
          self.paths[localID].outlet = True

      for p in self.paths:
        if not p.boundary:
          p.globalID = self.localPathID
          self.localToGlobal[p.localID] = self.localPathID
          self.localPathID += 1

      for p in self.paths:
        for s in p.sets:
          s.pathID = p.globalID

    def trim_paths(self):
      """
      Set set trim flag to True foall sets where path is not connected to inlet AND outlet
      """
      for p in self.paths:
        if not p.inlet and not p.outlet:
          for s in p.sets:
            s.trim = True
