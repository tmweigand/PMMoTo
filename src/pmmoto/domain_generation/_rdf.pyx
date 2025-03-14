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
from libcpp.memory cimport shared_ptr

from .rdf cimport generate_rdf

from .particles.atoms cimport AtomList
from ._particles cimport PyAtomList

__all__ = ["_generate_rdf"]

def _generate_rdf(probe_atoms, atoms, max_radius, bins, bin_width):
    """
    generate a radial distribution function 
    """
    cdef: 
        vector[long int] _bins

    _bins = bins
    bins = generate_rdf(
        (<PyAtomList>probe_atoms)._atom_list,
        (<PyAtomList>atoms)._atom_list,
        max_radius,
        _bins,
        bin_width
    )

    return bins