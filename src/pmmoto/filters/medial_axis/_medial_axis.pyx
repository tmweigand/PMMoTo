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

from ...core import octants

from .medial_extraction cimport coordinate
from .medial_extraction cimport get_neighborhood
from .medial_extraction cimport is_endpoint
from .medial_extraction cimport is_Euler_invariant
from .medial_extraction cimport is_simple_point
from .medial_extraction cimport find_simple_point_candidates
from .medial_extraction cimport find_boundary_simple_point_candidates
from .medial_extraction cimport is_last_boundary_point
from .medial_extraction cimport remove_points
__all__ = [
    "_get_neighborhood", 
    "_is_endpoint", 
    "_is_Euler_invariant", 
    "_is_simple_point",
    "_find_simple_point_candidates",
    "_find_boundary_simple_point_candidates",
    "_is_last_boundary_point",
    "_skeleton"
]


def _get_neighborhood(uint8_t[:,:,:] img, x, y, z, index = None):
    cdef:
        uint8_t* data_ptr = <uint8_t*>&img[0, 0, 0]
        size_t* shape_ptr = <size_t*>img.shape

    if index is None:
        index = [0,0,0]

    neighbors = get_neighborhood(data_ptr,x, y, z, shape_ptr, index)
    return np.array([neighbors[i] for i in range(27)], dtype=np.uint8)

def _is_endpoint(neighbors):
    return is_endpoint(neighbors)

def _is_Euler_invariant(neighbors, octants = None):
    if octants is None:
        octants = [0,1,2,3,4,5,6,7]
    return is_Euler_invariant(neighbors, octants)

def _is_simple_point(neighbors):
    return is_simple_point(neighbors)

def _is_last_boundary_point(neighbors,vertices):
    return is_last_boundary_point(neighbors,vertices)

def _find_boundary_simple_point_candidates(uint8_t[:, :, :] img, direction_index, loop, index, octants, vertices):


    cdef:
        uint8_t* data_ptr = &img[0, 0, 0]
        size_t* shape_ptr = <size_t*>img.shape
        vector[coordinate] coords
        vector[pair[int, int]] _loop


    for bounds in loop:
        _loop.push_back(pair[int, int](bounds[0], bounds[1]))


    find_boundary_simple_point_candidates(
        data_ptr,
        direction_index,
        coords,
        _loop,
        shape_ptr,
        index,
        octants,
        vertices
    )

    return coords


def _find_simple_point_candidates(uint8_t[:, :, :] img, direction_index, loop = None):

    cdef:
        uint8_t* data_ptr = &img[0, 0, 0]
        size_t* shape_ptr = <size_t*>img.shape
        vector[coordinate] coords
        vector[pair[int, int]] _loop

    for dim in range(3):
        _loop.push_back(pair[int, int](1, img.shape[dim] - 1))

    find_simple_point_candidates(
        data_ptr,
        direction_index,
        coords,
        _loop,
        shape_ptr,
    )

    return coords



def pm_find_simple_point_candidates(subdomain, img, dir_index):

    cdef list _coords = []
    cdef list coords

    for feature_id, feature in subdomain.features.all_features:
        feature_octants = octants.feature_octants(feature_id)
        vertices = feature.get_octant_vertices()
        feature_loop, _ = subdomain.features.get_feature_voxels(
            feature_id, extract_features=True
        )

        coords = _find_boundary_simple_point_candidates(
            img, dir_index, feature_loop, feature_id, feature_octants, vertices
        )
        _coords.extend(coords)

    _coords.extend(_find_simple_point_candidates(img, dir_index))

    return _coords

def _remove_points(uint8_t[:,:,:] img,coords):
    cdef:
        uint8_t* data_ptr = <uint8_t*>&img[0, 0, 0]
        size_t* shape_ptr = <size_t*>img.shape

    no_change  = remove_points(data_ptr,coords,shape_ptr)
    return no_change

def _skeleton(subdomain, img):

    num_directions = 6
    directions = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))

    converged = False
    while not converged:
        for dir_index in directions:
            coords = pm_find_simple_point_candidates(subdomain, img, dir_index)
            no_change = _remove_points(img,coords)
            if no_change:
                converged = True
