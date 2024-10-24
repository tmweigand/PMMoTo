# cython: profile=True
# cython: linetrace=True
# cython: boundscheck=False
# cython: wraparound=False
# from libcpp.vector cimport vector
# from numpy cimport uint64_t,int64_t
# from libc.stdio cimport printf

# TODO: Clean up type defs



# cdef inline uint64_t get_id(Py_ssize_t[3] x, int[3] voxels):
#     """
#     Determine the the id for a voxel. 
#     subdomain yields local id
#     domain yields global id
#     """
#     cdef int index_0, index_1, index_2

#     # Use modulo to handle periodic boundary conditions
#     index_0 = x[0] % voxels[0]
#     index_1 = x[1] % voxels[1]
#     index_2 = x[2] % voxels[2]

#     return index_0 * voxels[1] * voxels[2] + index_1 * voxels[2] + index_2

# cdef inline vector[int64_t] get_global_index(Py_ssize_t[3] x, int[3] domain_nodes, int[3] index_start):
#     """
#     Determine the global index [i,j,k]
#     """
#     cdef: 
#         int n
#         vector[int64_t] index

#     for n in range(0,3):
#         index.push_back(x[n] + index_start[n])
#     return index

# cdef inline vector[int64_t] get_global_index_periodic(Py_ssize_t[3] x, int[3] domain_nodes, int[3] index_start):
#     """
#     Determine the global index [i,j,k]. Loop around if periodic so match. 
#     """
#     cdef: 
#         int n
#         vector[int64_t] index

#     for n in range(0,3):
#         index.push_back(x[n] + index_start[n])
#         if index[n] >= domain_nodes[n]:
#             index[n] = 0
#         elif index[n] < 0:
#             index[n] = domain_nodes[n] - 1
#     return index