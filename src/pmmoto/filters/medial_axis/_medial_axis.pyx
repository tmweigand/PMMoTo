# cython: profile=True
# cython: linetrace=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
cnp.import_array()

from libc.stdint cimport uint8_t
from libcpp.vector cimport vector
from libcpp.pair cimport pair

from .medial_extraction cimport coordinate
from .medial_extraction cimport get_neighborhood
from .medial_extraction cimport is_endpoint
from .medial_extraction cimport is_Euler_invariant
from .medial_extraction cimport is_simple_point
from .medial_extraction cimport find_simple_points
from .medial_extraction cimport find_simple_point_candidates
from .medial_extraction cimport compute_thin_image

__all__ = [
    "_get_neighborhood", 
    "_is_endpoint", 
    "_is_Euler_invariant", 
    "_is_simple_point",
    "_find_simple_points",
    "_find_simple_point_candidates",
    "_compute_thin_image"
]


def _get_neighborhood(uint8_t[:,:,:] img, x, y, z, index = None):

    if index is None:
        index = [0,0,0]

    cdef:
        uint8_t* data_ptr = <uint8_t*>&img[0, 0, 0]
        size_t* shape_ptr = <size_t*>img.shape

    neighbors = get_neighborhood(data_ptr,x, y, z, shape_ptr, index)

    return np.array([neighbors[i] for i in range(27)], dtype=np.uint8)

def _is_endpoint(neighbors):
    return is_endpoint(neighbors)

def _is_Euler_invariant(neighbors, octants = None):
    if octants is None:
        octants = [0,1,2,3,4,5,6,7]
    return is_Euler_invariant(neighbors,octants)

def _is_simple_point(neighbors):
    return is_simple_point(neighbors)

def _find_simple_points(uint8_t[:, :, :] img, float[:, :, :] edt):
    cdef:
        uint8_t* data_ptr = &img[0, 0, 0]
        float* edt_ptr = &edt[0, 0, 0]
        size_t shape_ptr[3]
        vector[pair[int, int]] loop
        int dim

    # Copy shape from memoryview to shape_ptr
    for dim in range(3):
        shape_ptr[dim] = img.shape[dim]
        loop.push_back(pair[int, int](1, img.shape[dim] - 1))

    # Now call the C++ function with proper pointers
    coords = find_simple_points(data_ptr, edt_ptr, shape_ptr, loop)

    return coords


def _find_simple_point_candidates(uint8_t[:, :, :] img, int border, index = None):

    if index is None:
        index = [0,0,0]

    cdef:
        uint8_t* data_ptr = &img[0, 0, 0]
        size_t* shape_ptr = <size_t*>img.shape
        vector[coordinate] coords
        vector[pair[int, int]] loop

    for dim in range(3):
        loop.push_back(pair[int, int](1, img.shape[dim] - 1))

    find_simple_point_candidates(data_ptr,border,coords,loop,shape_ptr,index)

    return coords

def _compute_thin_image(uint8_t[:, :, :] img):
    cdef:
        uint8_t* data_ptr = &img[0, 0, 0]
        size_t* shape_ptr = <size_t*>img.shape


    compute_thin_image(data_ptr,shape_ptr)