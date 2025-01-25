import cython
import numpy as np
from numpy cimport npy_intp, uint64_t ,uint8_t, int64_t

from libcpp cimport bool
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp.algorithm cimport binary_search

cdef extern from "_voxels.hpp":

    cdef vector[ pair[unsigned long,unsigned long] ] unique_pairs(
        unsigned long* data,
        size_t nrows
    )

    cdef void loop_through_slice(
        uint8_t *segids,
        uint8_t *out,
        const int n,
        const long int stride,
        bool forward
        ) nogil

    cdef int64_t _get_nearest_boundary_index(
        uint8_t *img,
        uint8_t label,
        const int n,
        const long int stride,
        const int index_corrector,
        bool forward) nogil


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
