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


from .rdf cimport _generate_rdf

from .particles.atoms cimport Atom
from .particles.atoms cimport AtomList
from .particles.atoms cimport initialize_list

__all__ = ["generate_rdf"]

def generate_rdf(subdomain, atoms, probe_atoms, radius_in, num_bins):
    """
    generate a radial distribution function 
    """
    # build atom list
    # loop through probe_atom to collect neares
    # calculate the distance
    # bin distance counts
    cdef: 
        vector[double] point
        double radius = 0
        vector[vector[double]] atoms_c, probe_atoms_c
        vector[long int] rdf
        vector[vector[double]] box
        bool trim = False
        bool kd = True

    atoms_c = atoms
    probe_atoms_c = probe_atoms

    if subdomain is not None:
        point = subdomain.get_centroid()
        sd_radius = subdomain.get_radius()
        radius = sd_radius + radius_in
        trim = True
    else:
        point = [0,0,0]
        trim = False

    box = subdomain.own_box

    # cdef shared_ptr[AtomList] atom_list = initialize_list[AtomList,Atom](atoms_c,point,radius,box,kd,trim)
    # cdef shared_ptr[AtomList] probe_list = initialize_list[AtomList,Atom](probe_atoms_c,point,radius,box,kd,trim)
    
    # radius = radius_in
    # rdf = _generate_rdf(probe_list,atom_list,radius,num_bins)

    # cdef vector[vector[double]] check_atoms_c = return_particles(atom_list)

    # check_atoms = check_atoms_c

    return 0