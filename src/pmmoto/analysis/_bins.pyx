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
from libc.stdint cimport uint64_t

from .bins cimport count_locations, sum_masses

__all__ = ["_count_locations", "_sum_masses"]

def _count_locations(coordinates, dimension, num_bins, bin_width, min_bin_value):
    """
    generate a radial distribution function 
    """
    cdef: 
        vector[uint64_t] _bins
        vector[vector[double]] _coordinates

    _bins = vector[uint64_t](num_bins, 0)
    _coordinates = coordinates
    
    count_locations(
        _coordinates,
        dimension,
        _bins,
        bin_width,
        min_bin_value
    )

    return np.asarray(_bins)


def _sum_masses(coordinates, masses, dimension, num_bins, bin_width, min_bin_value):
    """
    sum masses 
    """
    cdef: 
        vector[double] _bins
        vector[vector[double]] _coordinates
        vector[double] _masses

    _bins = vector[double](num_bins, 0.0)
    _coordinates = coordinates
    _masses = masses
    
    sum_masses(
        _coordinates,
        _masses,
        dimension,
        _bins,
        bin_width,
        min_bin_value
    )

    return np.asarray(_bins)