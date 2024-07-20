# cython: profile=True
# cython: linetrace=True
import math
import numpy as np
cimport numpy as cnp
cimport cython
from mpi4py import MPI
comm = MPI.COMM_WORLD
from ..core import set

class medialSet(set.Set):
    def __init__(self, 
                 localID = 0, 
                 proc_ID = 0,
                 inlet = False, 
                 outlet = False, 
                 boundary = False, 
                 numNodes = 0, 
                 numBoundaryNodes = 0,
                 pathID = -1, 
                 type = 0, 
                 connectedNodes = None):
      super().__init__(localID, proc_ID, inlet, outlet, boundary, numNodes, numBoundaryNodes)
      self.pathID = pathID
      self.type = type
      self.connectedNodes = connectedNodes
      self.connectedSets = []
      self.globalConnectedSets = []
      self.trim = False
      self.inaccessible = False
      self.minDistance = math.inf
      self.maxDistance = -math.inf
      self.neighborProcID = []
      self.matchedSet = []
      self.connectedSets = []
      self.globalConnectedSets = []

    def getDistMinMax(self,data):
      for n in self.nodes:
        if data[n[0],n[1],n[2]] < self.minDistance:
          self.minDistance = data[n[0],n[1],n[2]]
        if data[n[0],n[1],n[2]] > self.maxDistance:
          self.maxDistance = data[n[0],n[1],n[2]]

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def getSetNodes(self,nNodes,indexMatch,nodeInfo,nodeInfoIndex):
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

      nodes = np.zeros([self.numNodes,3],dtype=np.uint64)
      cdef cnp.uint64_t [:,::1] _nodes
      _nodes = nodes

      bNodes = np.zeros([self.numBoundaryNodes,4],dtype=np.uint64)
      cdef cnp.uint64_t [:,::1] _bNodes
      _bNodes = bNodes

      boundaryFaces = np.zeros(26,dtype=np.uint8)
      cdef cnp.uint8_t [:] _boundaryFaces
      _boundaryFaces = boundaryFaces


      setNodes = self.numNodes
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

      self.setNodes(nodes)
      if bN > 0:
        self.setBoundaryNodes(bNodes,boundaryFaces)
