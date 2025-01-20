import cython

import numpy as np
from numpy cimport int8_t, int16_t, int32_t, int64_t
from numpy cimport uint8_t, uint16_t, uint32_t, uint64_t

from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "_distance.hpp":
    cdef inline void to_finite(
        float *f,
        const size_t voxels
    ) nogil

    cdef struct Hull:
        int vertex
        float height
        float range

    cdef void squared_edt_1d[T](
        T *labels,
        float *dest,
        int n,
        int stride,
        float anisotropy,
        const float lower_corrector,
        const float upper_corrector
    ) nogil

    cdef void squared_edt_1d_parabolic(
            float *img,
            const int n,
            const float resolution,
            const long int stride,
            vector[Hull] lower_hull,
            vector[Hull] upper_hull
        ) nogil

    cdef vector[Hull] return_boundary_hull(
        float *img,
        const int n,
        const float resolution,
        const long int stride,
        int num_hull,
        const int index_corrector,
        bool forward
        ) nogil