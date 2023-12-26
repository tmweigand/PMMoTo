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
cdef int[26][5] directions
cdef int numNeighbors
directions = cOrient.directions
numNeighbors = cOrient.numNeighbors

class Set(object):
    def __init__(self, 
                localID = 0, 
                proc_ID = 0,
                inlet = False, 
                outlet = False, 
                boundary = False, 
                numNodes = 0, 
                numBoundaryNodes = 0):    
      self.localID = localID   
      self.proc_ID = proc_ID
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
      
    def setNodes(self,nodes):
      self.nodes = nodes

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def get_set_nodes(self,nNodes,indexMatch,nodeInfo,nodeInfoIndex,subDomain):
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


    def setBoundaryNodes(self,boundaryNodes,boundaryFaces):
        self.boundary = True
        self.boundaryNodes = np.sort(boundaryNodes[:,0])
        self.boundaryNodeID = boundaryNodes[:,1:4]
        Orient = Orientation.Orientation()
        allFaces = Orient.allFaces
        for ID,bF in enumerate(boundaryFaces):
          if bF:
            faces = allFaces[ID]
            for f in faces:
              self.boundaryFaces[f] = 1
        self.numBoundaries = np.sum(self.boundaryFaces)