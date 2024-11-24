# cython: profile=True
# cython: linetrace=True
import math
import numpy as np
cimport numpy as cnp
cimport cython
from mpi4py import MPI
comm = MPI.COMM_WORLD

# from pmmoto.core import _set
# from pmmoto.core import _sets
# from pmmoto.core import orientation
# from pmmoto.core import _Orientation
# cOrient = _Orientation.cOrientation()
cdef int[26][5] directions
cdef int numNeighbors
# directions = cOrient.directions
# numNeighbors = cOrient.num_neighbors


# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)   # Deactivate negative indexing.
# def get_phase_boundary_nodes(grid,phaseID):
#   """
#   Loop through each face of the subDomain to determine the node closes to the boundary. 

#   Input: grid and phaseID

#   Output: list of nodes nearest boundary 
#   """
#   cdef cnp.uint8_t [:,:,:] _grid
#   _grid = grid

#   cdef int _phaseID = phaseID

#   grid_shape = np.array([grid.shape[0],grid.shape[1],grid.shape[2]],dtype=np.uint64)
#   cdef cnp.uint64_t [:] _grid_shape
#   _grid_shape = grid_shape

#   cdef int area
#   area = 2*_grid_shape[0]*_grid_shape[1] + 2*_grid_shape[0]*_grid_shape[2] + 2*_grid_shape[1]*_grid_shape[2]

#   solids = -np.ones([area,4],dtype=np.int32)
#   cdef cnp.int32_t [:,:] _solids
#   _solids = solids
  
#   order = np.ones((3), dtype=np.int32)
#   cdef cnp.int32_t [:] _order
#   _order = order

#   cdef int c,m,n,count,dir,numFaces,fIndex,solid
#   cdef int[3] arg_order

#   cdef int[6][4] face_info
#   face_info = cOrient.face_info

#   numFaces = cOrient.num_faces 
#   count = 0
#   for fIndex in range(0,numFaces):
#     dir = face_info[fIndex][3]
#     arg_order[0] = face_info[fIndex][0]
#     arg_order[1] = face_info[fIndex][1]
#     arg_order[2] = face_info[fIndex][2]

#     if dir == 1:
#       c_start = 0
#       c_end = _grid_shape[arg_order[0]]
#     else:
#       c_start = _grid_shape[arg_order[0]] - 1
#       c_end = 0

#     for m in range(0,_grid_shape[arg_order[1]]):
#       for n in range(0,_grid_shape[arg_order[2]]):
#         solid = False
#         c = c_start
#         while not solid and c != c_end:
#           _order[arg_order[0]] = c
#           _order[arg_order[1]] = m
#           _order[arg_order[2]] = n
#           if _grid[_order[0],_order[1],_order[2]] == _phaseID:
#             solid = True
#             _solids[count,0:3] = _order
#             _solids[count,3] = fIndex
#             count = count + 1
#           else:
#             c = c + dir
#         if (not solid and c == c_end):
#           _order[arg_order[0]] = -1
#           _solids[count,0:3] = _order
#           _solids[count,3] = fIndex
#           count = count + 1
  
#   return solids


# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)   # Deactivate negative indexing.
# def fixInterfaceCalc(tree,
#                       int lShape,
#                       int dir,
#                       cnp.ndarray[cnp.int32_t, ndim=2] _faceSolids,
#                       cnp.ndarray[cnp.float32_t, ndim=3] _EDT,
#                       cnp.ndarray[cnp.uint8_t, ndim=3] _visited,
#                       double min_dist,
#                       list coords,
#                       cnp.ndarray[cnp.uint8_t, ndim=1] argOrder):
#     """
#     Uses the solids from neighboring processes to determine if distance is less than determined
#     """
#     cdef int i,l,m,n,l_start,l_end,count,end_count
#     cdef float max_dist,d

#     _orderG = np.ones((1,3), dtype=np.double) #Global Order
    
#     _orderL = np.ones((3), dtype=np.uint32)   #Local Order
#     cdef cnp.uint32_t [:] orderL
#     orderL = _orderL

#     cdef cnp.double_t [:] c0 = coords[argOrder[0]]
#     cdef cnp.double_t [:] c1 = coords[argOrder[1]]
#     cdef cnp.double_t [:] c2 = coords[argOrder[2]]

#     if dir == 1:
#       l_start = 0
#       l_end = lShape
#     elif dir == -1:
#       l_start = lShape - 1
#       l_end = 0

#     for i in range(0,_faceSolids.shape[0]):

#         l = l_start
#         if _faceSolids[i,argOrder[0]] < 0:
#             end_count = np.abs(l_end-l)
#         else:
#             end_count = np.abs(_faceSolids[i,argOrder[0]]-l)

#         m = _faceSolids[i,argOrder[1]]
#         n = _faceSolids[i,argOrder[2]]
#         _orderG[0,argOrder[1]] = c1[m]
#         _orderG[0,argOrder[2]] = c2[n]
#         orderL[argOrder[1]] = m
#         orderL[argOrder[2]] = n

#         changed = True
#         count = 0
#         while changed and count < end_count:

#             _orderG[0,argOrder[0]] = c0[l]
#             orderL[argOrder[0]] = l
#             max_dist = _EDT[orderL[0],orderL[1],orderL[2]]
            
#             if (max_dist > min_dist):
#                 d,ind = tree.query(_orderG,distance_upper_bound = max_dist)
#                 if d < max_dist:
#                     _EDT[orderL[0],orderL[1],orderL[2]] = d
#                     changed = True
#                     _visited[orderL[0],orderL[1],orderL[2]] = 1

#                 elif _visited[orderL[0],orderL[1],orderL[2]] == 0:
#                     changed = False

#             l += dir
#             count += 1

#     return _EDT,_visited
