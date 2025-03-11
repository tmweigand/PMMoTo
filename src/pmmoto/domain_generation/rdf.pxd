"""rdf.pxd"""

from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr


from .particles.atoms cimport AtomList

cdef extern from "rdf.hpp":

    cdef vector[long int] generate_rdf(
        shared_ptr[AtomList] probe_atoms,
        shared_ptr[AtomList] atoms,
        double max_distance,
        vector[long int] bins,
        double bin_width
        ) 