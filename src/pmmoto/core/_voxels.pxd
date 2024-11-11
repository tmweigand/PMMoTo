from libcpp.vector cimport vector
from numpy cimport npy_intp, npy_int8, npy_uint8, ndarray, npy_float32
from libcpp cimport bool
from libcpp.utility cimport pair
from libcpp.algorithm cimport binary_search

cdef inline  npy_intp mod(npy_intp a, npy_intp base):    
  return ((a % base) + base) % base;


cdef inline bool _match_boundary_voxels(vector[npy_intp] list1, vector[npy_intp] list2):
    """
    Input: Two Sorted Lists
    Output: Bool of at least one shared element
    """
    cdef bool match = False
    for l in list1:
        if (binary_search(list2.begin(), list2.end(), l)):
            match = True
            break
    return match

cdef inline int count_matched_voxels(vector[npy_intp] list1, vector[npy_intp] list2):
    cdef int count = 0
    for l in list1:
        if (binary_search(list2.begin(), list2.end(), l)):
            count += 1
    return count

# cdef struct match:
#     npy_intp id
#     npy_intp global_id
#     npy_intp rank
#     vector[npy_intp] neighbor_id
#     vector[npy_intp] neighbor_rank