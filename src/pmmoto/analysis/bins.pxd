"""rdf.pxd"""

from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libc.stdint cimport uint64_t

from ..particles.atoms cimport AtomList

cdef extern from "bins.hpp":

    cdef void count_locations(
        vector[vector[double]] atoms,
        int dimension,
        vector[uint64_t] bins,
        double bin_width,
        double min_bin_value
        ) 

    cdef void sum_masses(
        vector[vector[double]] atoms,
        vector[double] masses,
        int dimension,
        vector[double] bins,
        double bin_width,
        double min_bin_value
        ) 