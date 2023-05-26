cimport cython
from libcpp.vector cimport vector
import numpy as np
from numpy cimport npy_intp, npy_uint8, ndarray
from libcpp cimport bool


ctypedef npy_uint8 pixel_type

# struct to hold 3D coordinates
cdef struct coordinate:
    npy_intp p
    npy_intp r
    npy_intp c
    npy_intp ID
    npy_intp faceCount

cdef bool compare(coordinate l, const coordinate r) nogil

cdef void find_simple_point_candidates(pixel_type[:, :, ::1] img,
                                       int curr_border,
                                       vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_boundary(pixel_type[:, :, ::1] img,
                                        int curr_border,
                                        vector[coordinate] & simple_border_points) nogil


cdef void get_neighborhood(pixel_type[:, :, ::1] img,
                           npy_intp p, npy_intp r, npy_intp c,
                           pixel_type neighborhood[]) nogil

cdef void get_neighborhood_limited(pixel_type[:, :, ::1] img,
                               npy_intp p, npy_intp r, npy_intp c, npy_intp ID,
                               pixel_type neighborhood[]) nogil

cdef bint is_simple_point(pixel_type neighbors[]) nogil

cdef bint is_surface_point(pixel_type neighbors[]) nogil

cdef int is_endpoint_check(pixel_type neighbors[]) nogil