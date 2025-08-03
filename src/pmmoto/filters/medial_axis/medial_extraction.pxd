"""medial_extraction.pxd"""

from libc.stdint cimport uint8_t
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.pair cimport pair

cdef extern from "medial_extraction.hpp":
    
    struct coordinate:
        int x
        int y
        int z
        vector[int] index
        vector[int] vertices
        float edt
    
    vector[uint8_t] get_neighborhood(const uint8_t* img,
                 int x,
                 int y,
                 int z,
                 const size_t* strides,
                 const vector[int]& index)

    bool is_endpoint(vector[uint8_t] neighbors)

    bool is_last_boundary_point(
        const vector[uint8_t]& neighbors, 
        const vector[int]& vertices)

    bool is_Euler_invariant(
        vector[uint8_t] neighbors,
        const vector[int]& octants)

    bool is_simple_point(vector[uint8_t] neighbors)

    void find_boundary_simple_point_candidates(
        uint8_t* img,
        const vector[int]& erode_index,
        vector[coordinate] simple_border_points,
        const vector[pair[int, int]] loop,
        const size_t* shape,
        const vector[int]& index,
        const vector[int]& octants,
        const vector[int]& vertices)

    bool remove_points(
        uint8_t* img,
        vector[coordinate]& simple_points,
        const size_t* shape,
    )

    void find_simple_point_candidates(
        uint8_t* img,
        const vector[int]& erode_index,
        vector[coordinate] simple_border_points,
        const vector[pair[int, int]] loop,
        const size_t* shape)