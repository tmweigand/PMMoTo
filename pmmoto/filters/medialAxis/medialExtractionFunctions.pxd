cimport cython
from libcpp.vector cimport vector
import numpy as np
from numpy cimport npy_intp, npy_int8, npy_uint8, ndarray, npy_float32
from libcpp cimport bool

ctypedef npy_uint8 pixel_type

# struct to hold 3D coordinates
cdef struct coordinate:
    npy_intp x
    npy_intp y
    npy_intp z
    npy_intp ID
    npy_intp faceCount
    npy_float32 edt

cdef inline bool compare(coordinate l, const coordinate r):
    return l.c > r.c;

cdef void findSimplePoints(pixel_type[:, :, ::1] img,
                           npy_float32[:, :, ::1] edt,
                           int fErode,
                           npy_intp[:,:] fLoop,
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


cdef void find_simple_point_candidates_faces_0(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_faces_1(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_faces_2(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_faces_3(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_faces_4(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_faces_5(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil


cdef void find_simple_point_candidates_edges_0(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_edges_1(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_edges_2(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil
                            
cdef void find_simple_point_candidates_edges_3(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_edges_4(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_edges_5(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_edges_6(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil
                        
cdef void find_simple_point_candidates_edges_7(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_edges_8(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_edges_9(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_edges_10(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil            
        
cdef void find_simple_point_candidates_edges_11(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil   

cdef void find_simple_point_candidates_corners_0(pixel_type[:, :, ::1] img,
                                                npy_float32[:, :, ::1] edt,
                                                int curr_border,
                                                npy_intp [:,:] fLoop,
                                                vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_corners_1(pixel_type[:, :, ::1] img,
                                                npy_float32[:, :, ::1] edt,
                                                int curr_border,
                                                npy_intp [:,:] fLoop,
                                                vector[coordinate] & simple_border_points) nogil                            

cdef void find_simple_point_candidates_corners_2(pixel_type[:, :, ::1] img,
                                                npy_float32[:, :, ::1] edt,
                                                int curr_border,
                                                npy_intp [:,:] fLoop,
                                                vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_corners_3(pixel_type[:, :, ::1] img,
                                                npy_float32[:, :, ::1] edt,
                                                int curr_border,
                                                npy_intp [:,:] fLoop,
                                                vector[coordinate] & simple_border_points) nogil                                                

cdef void find_simple_point_candidates_corners_4(pixel_type[:, :, ::1] img,
                                                npy_float32[:, :, ::1] edt,
                                                int curr_border,
                                                npy_intp [:,:] fLoop,
                                                vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_corners_5(pixel_type[:, :, ::1] img,
                                                npy_float32[:, :, ::1] edt,
                                                int curr_border,
                                                npy_intp [:,:] fLoop,
                                                vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_corners_6(pixel_type[:, :, ::1] img,
                                                npy_float32[:, :, ::1] edt,
                                                int curr_border,
                                                npy_intp [:,:] fLoop,
                                                vector[coordinate] & simple_border_points) nogil

cdef void find_simple_point_candidates_corners_7(pixel_type[:, :, ::1] img,
                                                npy_float32[:, :, ::1] edt,
                                                int curr_border,
                                                npy_intp [:,:] fLoop,
                                                vector[coordinate] & simple_border_points) nogil