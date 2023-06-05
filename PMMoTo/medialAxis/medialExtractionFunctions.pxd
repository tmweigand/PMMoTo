cimport cython
from libcpp.vector cimport vector
import numpy as np
from numpy cimport npy_intp, npy_uint8, ndarray
from libcpp cimport bool


ctypedef npy_uint8 pixel_type

# struct to hold 3D coordinates
cdef struct coordinate:
    npy_intp x
    npy_intp y
    npy_intp z
    npy_intp ID
    npy_intp faceCount

cdef struct coordinateInfo:
    npy_intp x
    npy_intp y
    npy_intp z
    npy_intp ID
    npy_intp faceCount
    npy_intp change

cdef inline bool  compare(coordinate l, const coordinate r):
    return l.c > r.c;

cdef void find_simple_point_candidates(pixel_type[:, :, ::1] img,
                                       int curr_border,
                                       vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_boundary(pixel_type[:, :, ::1] img,
                                        int curr_border,
                                        vector[coordinate] & simple_border_points) nogil


cdef void get_neighborhood(pixel_type[:, :, ::1] img,
                           npy_intp x, npy_intp y, npy_intp z,
                           pixel_type neighborhood[]) nogil

cdef void get_neighborhood_limited(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z, npy_intp ID,
                               pixel_type neighborhood[]) nogil

cdef bint is_simple_point(pixel_type neighbors[]) nogil

cdef bint is_surface_point(pixel_type neighbors[]) nogil

cdef int is_endpoint_check(pixel_type neighbors[]) nogil

cdef void find_simple_point_candidates_internalfaces_0(pixel_type[:, :, ::1] img,
                                        vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_internalfaces_1(pixel_type[:, :, ::1] img,
                                        vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_internalfaces_2(pixel_type[:, :, ::1] img,
                                        vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_internalfaces_3(pixel_type[:, :, ::1] img,
                                        vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_internalfaces_4(pixel_type[:, :, ::1] img,
                                        vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_internalfaces_5(pixel_type[:, :, ::1] img,
                                        vector[coordinate] & simple_border_points) nogil