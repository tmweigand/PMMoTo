from libcpp.vector cimport vector
from numpy cimport npy_intp, npy_int8, npy_uint8, ndarray, npy_float32
from libcpp cimport bool
from libcpp.utility cimport pair
from libcpp.algorithm cimport binary_search

cdef inline bool match_boundary_nodes(vector[npy_intp] list1, vector[npy_intp] list2):
    """
    Input: Two Sorted Lists
    Output: Bool of at least one shared element
    """
    cdef bool match =  False
    for l in list1:
        if (binary_search(list2.begin(), list2.end(), l)):
            match = True
            break
    return match