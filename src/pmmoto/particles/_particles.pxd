"""_particles.pxd"""
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

from .atoms cimport AtomList
from .spheres cimport SphereList
from .cylinders cimport CylinderList

cdef class PyAtomList:
    cdef shared_ptr[AtomList] _atom_list
    
cdef class PySphereList:
    cdef shared_ptr[SphereList] _sphere_list

cdef class PyCylinderList:
    cdef shared_ptr[CylinderList] _cylinder_list