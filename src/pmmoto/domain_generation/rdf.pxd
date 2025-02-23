"""rdf.pxd"""

from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr


from .atoms cimport AtomList

cdef extern from "rdf.hpp":
    cdef vector[long int] _generate_rdf(
        shared_ptr[AtomList] probe_atoms,
        shared_ptr[AtomList] atoms,
        double radius,
        int num_bins
        ) 