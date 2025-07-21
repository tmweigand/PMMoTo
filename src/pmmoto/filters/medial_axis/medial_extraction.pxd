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
        int ID
        int faceCount
        float edt
    
    vector[uint8_t] get_neighborhood(const uint8_t* img,
                 int x,
                 int y,
                 int z,
                 const size_t* strides,
                 const vector[int]& index)
    bool is_endpoint(vector[uint8_t] neighbors)
    bool is_Euler_invariant(
        vector[uint8_t] neighbors,
        const vector[int]& octants)
    bool is_simple_point(vector[uint8_t] neighbors)
    vector[coordinate] find_simple_points(
        const uint8_t* img,
        const float* edt,
        const size_t* shape,
        const vector[pair[int, int]] loop)
    void find_simple_point_candidates(
        uint8_t* img,
        int curr_border,
        vector[coordinate] simple_border_points,
        const vector[pair[int, int]] loop,
        const size_t* shape,
        const vector[int]& offset)
    void compute_thin_image(
        uint8_t* img, 
        const size_t* shape)