# cython: profile=True
# cython: linetrace=True
import math
import numpy as np
cimport numpy as cnp
cimport cython
from mpi4py import MPI
comm = MPI.COMM_WORLD

from . import set
from . import sets
from . import Orientation
cOrient = Orientation.cOrientation()
cdef int[26][5] directions
cdef int numNeighbors
directions = cOrient.directions
numNeighbors = cOrient.numNeighbors


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_node_info(rank,grid,phase,inlet,outlet,Domain,loopInfo,subDomain,Orientation):
  """
  Gather information for the nodes. Loop through internal nodes first and
  then go through boundaries.

  Input: Binary grid and Domain,Subdomain,Orientation information,indexOrder

  IndexOrder arranges the ordering of loopInfo so inlet and outlet faces 
  are first to ensure optimal looping and correct boundary values

  Output:
  nodeInfo: [boundary,inlet,outlet,boundaryID,availDirection,lastDirection,visited]
  nodeInfoIndex:[i,j,k,globalIndex,global i,global j,global k]
  nodeDirections: availble directions[26]
  nodeDirectionsIndex: index of availble directions[26]
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

  nodeTable = -np.ones_like(grid,dtype=np.uint64)
  cdef cnp.uint64_t [:,:,:] _nodeTable
  _nodeTable = nodeTable

  cdef int c,d,i,j,k,ii,jj,kk,availDirection,sInlet,sOutlet
  cdef int iLoc,jLoc,kLoc,globIndex
  cdef int iMin,iMax,jMin,jMax,kMin,kMax
  cdef int _phase = phase

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

  if phase == 0:
    gridP = np.pad(grid,1,constant_values=1)
  else:
    gridP = np.pad(grid,1,constant_values=0)
  cdef cnp.uint8_t [:,:,:] _ind
  _ind = gridP

  cdef cnp.int64_t [:,:,:] _loopInfo
  _loopInfo = loopInfo

  cdef int dN0,dN1,dN2
  dN0 = Domain.nodes[0]
  dN1 = Domain.nodes[1]
  dN2 = Domain.nodes[2]

  # Loop through Boundary Faces to get nodeInfo and nodeIndex
  c = 0
  for fIndex in range(0,numFaces):
    iMin = _loopInfo[fIndex][0][0]
    iMax = _loopInfo[fIndex][0][1]
    jMin = _loopInfo[fIndex][1][0]
    jMax = _loopInfo[fIndex][1][1]
    kMin = _loopInfo[fIndex][2][0]
    kMax = _loopInfo[fIndex][2][1]
    bID = np.asarray(Orientation.faces[fIndex]['ID'],dtype=np.int8)
    sInlet = inlet[fIndex]
    sOutlet = outlet[fIndex]
    for i in range(iMin,iMax):
      for j in range(jMin,jMax):
        for k in range(kMin,kMax):
          if _ind[i+1,j+1,k+1] == _phase:

            iLoc = iStart + i
            jLoc = jStart + j
            kLoc = kStart + k

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

            boundID = cOrient.getBoundaryIDReference(boundaryID)
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
  iMin = _loopInfo[numFaces][0][0]
  iMax = _loopInfo[numFaces][0][1]
  jMin = _loopInfo[numFaces][1][0]
  jMax = _loopInfo[numFaces][1][1]
  kMin = _loopInfo[numFaces][2][0]
  kMax = _loopInfo[numFaces][2][1]
  for i in range(iMin,iMax):
    for j in range(jMin,jMax):
      for k in range(kMin,kMax):
        if (_ind[i+1,j+1,k+1] == _phase):
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
    iMin = _loopInfo[fIndex][0][0]
    iMax = _loopInfo[fIndex][0][1]
    jMin = _loopInfo[fIndex][1][0]
    jMax = _loopInfo[fIndex][1][1]
    kMin = _loopInfo[fIndex][2][0]
    kMax = _loopInfo[fIndex][2][1]
    for i in range(iMin,iMax):
      for j in range(jMin,jMax):
        for k in range(kMin,kMax):
          if _ind[i+1,j+1,k+1] == _phase:
            availDirection = 0
            for d in range(0,numNeighbors):
              ii = directions[d][0]
              jj = directions[d][1]
              kk = directions[d][2]
              if (_ind[i+ii+1,j+jj+1,k+kk+1] == _phase):
                node = _nodeTable[i+ii,j+jj,k+kk]
                _nodeDirections[c,d] = 1
                _nodeDirectionsIndex[c,d] = node
                availDirection += 1

            _nodeInfo[c,4] = availDirection
            c = c + 1

  # Loop through internal nodes to get nodeDirections and _nodeDirectionsIndex
  iMin = _loopInfo[numFaces][0][0]
  iMax = _loopInfo[numFaces][0][1]
  jMin = _loopInfo[numFaces][1][0]
  jMax = _loopInfo[numFaces][1][1]
  kMin = _loopInfo[numFaces][2][0]
  kMax = _loopInfo[numFaces][2][1]
  for i in range(iMin,iMax):
   for j in range(jMin,jMax):
     for k in range(kMin,kMax):
       if _ind[i+1,j+1,k+1] == _phase:
         availDirection = 0
         for d in range(0,numNeighbors):
           ii = directions[d][0]
           jj = directions[d][1]
           kk = directions[d][2]
           if (_ind[i+ii+1,j+jj+1,k+kk+1] == _phase):
             node = _nodeTable[i+ii,j+jj,k+kk]
             _nodeDirections[c,d] = 1
             _nodeDirectionsIndex[c,d] = node
             availDirection += 1

         _nodeInfo[c,4] = availDirection
         c = c + 1

  return [nodeInfo,nodeInfoIndex,nodeDirections,nodeDirectionsIndex]


cdef get_node_boundary_info(cnp.int8_t[:] cNode,int numBNodes,bint sBound,bint sInlet,bint sOutlet):
  """
  Get Node Info about Boundary, Inlet, Outlet
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
def get_connected_sets(subDomain,grid,phaseID,Nodes):
  """
  Connects the NxNxN (or NXN) nodes into connected sets.
  1. Inlet
  2. Outlet
  """
  cdef int node,ID,nodeValue,d,oppDir,rank
  cdef int numNodes,numSetNodes,numNodesPhase,setCount

  rank = subDomain.ID

  numNodesPhase = np.count_nonzero(grid == phaseID)

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
          numBNodes,sBound,sInlet,sOutlet = get_node_boundary_info(cNode,numBNodes,sBound,sInlet,sOutlet)

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
        Sets.append(set.Set(localID = setCount,
                                  proc_ID = rank,
                                  inlet = sInlet,
                                  outlet = sOutlet,
                                  boundary = sBound,
                                  numNodes = numSetNodes,
                                  numBoundaryNodes = numBNodes))

        Sets[setCount].get_set_nodes(numNodes,indexMatch,_nodeInfo,_nodeInfoIndex,subDomain)
        setCount = setCount + 1

      numSetNodes = 0
      numBNodes = 0

  allSets = sets.Sets(Sets,setCount,subDomain)

  return allSets

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_boundary_nodes(grid,phaseID):
  """
  Loop through each face of the subDomain to determine the node closes to the boundary. 

  Input: grid and phaseID

  Output: list of nodes nearest boundary 
  """
  cdef cnp.uint8_t [:,:,:] _grid
  _grid = grid

  cdef int _phaseID = phaseID

  grid_shape = np.array([grid.shape[0],grid.shape[1],grid.shape[2]],dtype=np.uint64)
  cdef cnp.uint64_t [:] _grid_shape
  _grid_shape = grid_shape

  cdef int area
  area = 2*_grid_shape[0]*_grid_shape[1] + 2*_grid_shape[0]*_grid_shape[2] + 2*_grid_shape[1]*_grid_shape[2]

  solids = -np.ones([area,4],dtype=np.int32)
  cdef cnp.int32_t [:,:] _solids
  _solids = solids
  
  order = np.ones((3), dtype=np.int32)
  cdef cnp.int32_t [:] _order
  _order = order

  cdef int c,m,n,count,dir,numFaces,fIndex,solid
  cdef int[3] arg_order

  cdef int[6][4] face_info
  face_info = cOrient.face_info

  numFaces = cOrient.numFaces 
  count = 0
  for fIndex in range(0,numFaces):
    dir = face_info[fIndex][3]
    arg_order[0] = face_info[fIndex][0]
    arg_order[1] = face_info[fIndex][1]
    arg_order[2] = face_info[fIndex][2]

    if (dir == 1):
        for m in range(0,_grid_shape[arg_order[1]]):
            for n in range(0,_grid_shape[arg_order[2]]):
                solid = False
                c = 0
                while not solid and c < _grid_shape[arg_order[0]]:
                    _order[arg_order[0]] = c
                    _order[arg_order[1]] = m
                    _order[arg_order[2]] = n
                    if _grid[_order[0],_order[1],_order[2]] == _phaseID:
                        solid = True
                        _solids[count,0:3] = _order
                        _solids[count,3] = fIndex
                        count = count + 1
                    else:
                        c = c + 1
                if (not solid and c == _grid_shape[arg_order[0]]):
                    _order[arg_order[0]] = -1
                    _solids[count,0:3] = _order
                    _solids[count,3] = fIndex
                    count = count + 1

    elif (dir == -1):
        for m in range(0,_grid_shape[arg_order[1]]):
            for n in range(0,_grid_shape[arg_order[2]]):
                solid = False
                c = _grid_shape[arg_order[0]] - 1
                while not solid and c > 0:
                    _order[arg_order[0]] = c
                    _order[arg_order[1]] = m
                    _order[arg_order[2]] = n
                    if _grid[_order[0],_order[1],_order[2]] == _phaseID:
                        solid = True
                        _solids[count,0:3] = _order
                        _solids[count,3] = fIndex
                        count = count + 1
                    else:
                        c = c - 1
                if (not solid and c == 0):
                    _order[arg_order[0]] = -1
                    _solids[count,0:3] = _order
                    _solids[count,3] = fIndex
                    count = count + 1
  
  return solids