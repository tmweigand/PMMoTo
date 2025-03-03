# cython: profile=True
# cython: linetrace=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
cnp.import_array()

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

from .particles.atoms cimport Atom,AtomList,initialize_list,return_particles,set_own_particles


__all__ = ["initialize"]

cdef class PyAtomList:
    cdef shared_ptr[AtomList] ptr  # Store the shared pointer

    @staticmethod
    cdef PyAtomList from_ptr(shared_ptr[AtomList] atom_list):
        """
        Create a PyAtomList from a shared_ptr[AtomList].
        """
        cdef PyAtomList obj = PyAtomList.__new__(PyAtomList)
        obj.ptr = atom_list  # Assign the shared pointer
        return obj
    
    def return_np_array(self, return_own = False):
        cdef vector[vector[double]] particles = return_particles[AtomList](self.ptr,return_own)
        return(np.asarray(particles))


def initialize(subdomain, particles, add_periodic, set_own):
    """
    Initialize a list of particles (i.e. atoms, spheres).
    Particles must be a np array of size (n_atoms,4)=>(x,y,z,radius)
    If add_periodic: particles that cross the domain boundary will be add.
    If trim: particles that do not cross the subdomain boundary will be deleted.
    """
    cdef: 
        vector[vector[double]] _particles
        vector[vector[double]] domain_box,subdomain_box,subdomain_own_box
        bool _add_periodic

    _particles = particles
    domain_box = subdomain.domain.box
    subdomain_box = subdomain.box
    subdomain_own_box = subdomain.own_box
    _add_periodic = add_periodic

    cdef shared_ptr[AtomList] atom_list = initialize_list[AtomList,Atom](_particles,domain_box,subdomain_box,_add_periodic)

    if set_own:
        set_own_particles[AtomList](atom_list,subdomain_own_box)

    return PyAtomList.from_ptr(atom_list)

