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


from .bins cimport count_locations, sum_masses

__all__ = ["_count_locations", "_sum_masses"]

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


def _sum_masses(coordinates, masses, dimension, bins, bin_width, min_bin_value):
    """
    sum masses 
    """
    cdef: 
        vector[double] _bins
        vector[vector[double]] _coordinates
        vector[double] _masses

    _bins = bins
    _coordinates = coordinates
    _masses = masses
    
    bins = sum_masses(
        _coordinates,
        _masses,
        dimension,
        _bins,
        bin_width,
        min_bin_value
    )

    return np.asarray(bins)