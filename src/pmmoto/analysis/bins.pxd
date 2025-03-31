"""rdf.pxd"""

from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr


from ..particles.atoms cimport AtomList

cdef extern from "bins.hpp":

    cdef vector[unsigned long long] count_locations(
        vector[vector[double]] atoms,
        int dimension,
        vector[unsigned long long] bins,
        double bin_width,
        double min_bin_value
        ) 

    cdef vector[double] sum_masses(
        vector[vector[double]] atoms,
        vector[double] masses,
        int dimension,
        vector[double] bins,
        double bin_width,
        double min_bin_value
        ) 