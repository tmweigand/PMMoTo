"""_particles.pxd"""
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

from .atoms cimport AtomList
from .spheres cimport SphereList

cdef class PyAtomList:
    cdef shared_ptr[AtomList] _atom_list
    

cdef class PySphereList:
    cdef shared_ptr[SphereList] _sphere_list
    