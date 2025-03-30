# cython: profile=True
# cython: linetrace=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
from libcpp cimport bool
from libcpp.vector cimport vector


from .bins cimport count_locations

__all__ = ["_count_locations"]

def _count_locations(coordinates, dimension, bins, bin_width, min_bin_value):
    """
    generate a radial distribution function 
    """
    cdef: 
        vector[unsigned long long] _bins
        vector[vector[double]] _coordinates

    _bins = bins
    _coordinates = coordinates
    bins = count_locations(
        _coordinates,
        dimension,
        _bins,
        bin_width,
        min_bin_value
    )

    return np.asarray(bins)