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
from libcpp.unordered_map cimport unordered_map

from .spheres cimport SphereList
from .particle_list cimport Box
from .atoms cimport AtomList,atom_id_to_radius,group_atoms_by_type


__all__ = ["_initialize_atoms","_initialize_spheres"]


def create_box(bounds):
    """
    Convert a tuple of tuples ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    into a C++ Box structure in Cython.
    """
    cdef Box box
    
    # Directly assign values from the Python tuple
    box.min[0], box.max[0] = bounds[0]
    box.min[1], box.max[1] = bounds[1]
    box.min[2], box.max[2] = bounds[2]

    return box


class AtomMap():
    """
    Wrapper to a map of atom lists what are broken down by atom type
    """

    def __init__(self, atom_map, labels):
        self.atom_map = atom_map 
        self.labels = labels

    def return_np_array(self,return_own = False, return_label = False):
        """
        Return atoms as np.array
        """
        cdef vector[vector[double]] particles 
        particle_list = [] 
        for label in self.labels:
            particles = self.atom_map[label].return_np_array(return_own, return_label)
            particle_list.append(np.asarray(particles))

        return np.concatenate(particle_list, axis=0)

    def size(self):
        """
        Add the periodic atoms for each atom type
        """
        atom_counts = {}
        for label in self.labels:
            atom_counts[label] = self.atom_map[label].size()

        return atom_counts

    def add_periodic(self,subdomain):
        """
        Add the periodic atoms for each atom type
        """
        for label in self.labels:
            self.atom_map[label].add_periodic(subdomain)
    

    def set_own(self,subdomain):
        """
        Determine which atoms owned by this mpi process
        """
        for label in self.labels:
            self.atom_map[label].set_own(subdomain)  


    def trim_intersecting(self,subdomain):
        """
        Remove all atoms that are not within nor intersect box
        """
        for label in self.labels:
            self.atom_map[label].trim_intersecting(subdomain)   


    def trim_within(self,subdomain):
        """
        Remove all atoms that are not within box
        """
        for label in self.labels:
            self.atom_map[label].trim_within(subdomain)   

    def return_list(self,label):
        """
        Return the AtomList of the label
        """
        if label not in self.atom_map:
            raise KeyError(f"Label {label} not found in atom map")
        return self.atom_map[label]


cdef class PyAtomList:
    
    def __cinit__(self, vector[vector[double]] coordinates, double radius, int label):
        """
        Initialize a PyAtomList that contains a shared point to the c++ implementation
        """
        if coordinates.size() == 0:
            raise ValueError("Cannot initialize PyAtomList with empty coordinates")
            
        try:
            self._atom_list = shared_ptr[AtomList](new AtomList(coordinates, radius, label))
            if not self._atom_list:
                raise RuntimeError("Failed to initialize AtomList")
        except Exception as e:
            raise RuntimeError(f"Failed to create AtomList: {str(e)}")

    def __dealloc__(self):
        """Ensure proper cleanup of resources"""
        if self._atom_list != NULL: # Add NULL check
            self._atom_list.reset()
            self._atom_list = shared_ptr[AtomList]()

    @property
    def radius(self):
        return self._atom_list.get().radius


    def size(self):
        return self._atom_list.get().size()

    def build_KDtree(self):
        """
        Build a KD tree from the coordinates
        """
        self._atom_list.get().build_KDtree()

    def return_np_array(self, return_own = False, return_label = False):
        """
        Return atoms as np array
        """
        cdef vector[vector[double]] particles = self._atom_list.get().return_atoms(return_own, return_label)

        return(np.asarray(particles))

    def add_periodic(self, subdomain):
        """
        Add the periodic atoms for each atom type
        """
        cdef Box _domain_box, _subdomain_box

        _domain_box = create_box(subdomain.domain.box)
        _subdomain_box = create_box(subdomain.box)

        self._atom_list.get().add_periodic_atoms(_domain_box, _subdomain_box)

    def set_own(self, subdomain):
        """
        Determine which atoms owned by this mpi process
        """
        cdef Box _subdomain_box
        _subdomain_box = create_box(subdomain.own_box)

        self._atom_list.get().set_own_atoms(_subdomain_box) 

    def trim_intersecting(self, subdomain):
        """
        Remove all atoms that are not within nor intersect box
        """
        cdef Box _subdomain_box
        _subdomain_box = create_box(subdomain.box)

        self._atom_list.get().trim_atoms_intersecting(_subdomain_box)  


    def trim_within(self, subdomain):
        """
        Remove all atoms that are not within box
        """
        cdef Box _subdomain_box
        _subdomain_box = create_box(subdomain.own_box)

        self._atom_list.get().trim_atoms_within(_subdomain_box)  


cdef class PySphereList:

    def __cinit__(self, vector[vector[double]] coordinates, vector[double] radii):
        """
        Initialize a PySphereList
        """
        if coordinates.size() == 0:
            raise ValueError("Cannot initialize PySphereList with empty coordinates")
        try:
            self._sphere_list = shared_ptr[SphereList](new SphereList(coordinates, radii))
            if not self._sphere_list:
                raise RuntimeError("Failed to initialize SphereList")
        except Exception as e:
            raise RuntimeError(f"Failed to create SphereList: {str(e)}")

    def size(self):
        return self._sphere_list.get().size()

    def build_KDtree(self):
        """
        Build a KD tree from the coordinates
        """
        self._sphere_list.get().build_KDtree()
        

    def return_np_array(self, return_own = False):
        """
        Return atoms as np array
        """
        cdef vector[vector[double]] particles = self._sphere_list.get().return_spheres(return_own)

        return(np.asarray(particles))

    def add_periodic(self,subdomain):
        """
        Add the periodic atoms for each atom type
        """
        cdef Box _domain_box,_subdomain_box

        _domain_box = create_box(subdomain.domain.box)
        _subdomain_box = create_box(subdomain.box)

        self._sphere_list.get().add_periodic_spheres(_domain_box,_subdomain_box)

    def set_own(self,subdomain):
        """
        Determine which atoms owned by this mpi process
        """
        cdef Box _subdomain_box
        _subdomain_box = create_box(subdomain.own_box)

        self._sphere_list.get().own_spheres(_subdomain_box) 

    def trim_intersecting(self,subdomain):
        """
        Remove all spheres that are not within nor intersect box
        """
        cdef Box _subdomain_box
        _subdomain_box = create_box(subdomain.box)

        self._sphere_list.get().trim_spheres_intersecting(_subdomain_box)  

    def trim_within(self,subdomain):
        """
        Remove all spheres that are not within box
        """
        cdef Box _subdomain_box
        _subdomain_box = create_box(subdomain.own_box)

        self._sphere_list.get().trim_spheres_within(_subdomain_box)  


def _initialize_atoms(atom_coordinates, atom_radii, atom_ids, by_type = False):
    """
    Initialize a list of atoms. 
    """

    cdef vector[double] radii
    
    if by_type:
        return _initialize_atoms_by_type(atom_coordinates, atom_radii, atom_ids)
        
    else:
        radii = atom_id_to_radius(atom_ids,atom_radii)
        return _initialize_spheres(atom_coordinates, radii)

def _initialize_atoms_by_type(atom_coordinates, atom_radii, atom_ids):
    """
    Initialize a list of particles (i.e. atoms, spheres).
    Particles must be a np array of size (n_atoms,4)=>(x,y,z,radius)
    If add_periodic: particles that cross the domain boundary will be add.
    If trim: particles that do not cross the subdomain boundary will be deleted.
    """
    cdef: 
        vector[vector[double]] _atom_coordinates
        vector[int] _atom_ids
        unordered_map[int, double] _radii

    labels = np.unique(atom_ids)

    _atom_coordinates = atom_coordinates
    _atom_ids = atom_ids
    _radii = atom_radii

    cdef unordered_map[int,vector[vector[double]]] atom_groups = group_atoms_by_type(_atom_coordinates,_atom_ids)

    atom_lists = {}
    for label in labels:
        atom_lists[label] = PyAtomList(atom_groups[label], _radii[label], label)
    
    return AtomMap(atom_lists,labels)

def _initialize_spheres(spheres, radii):
    """
    Initialize a list of particles (i.e. atoms, spheres).
    Particles must be a np array of size (n_atoms,4)=>(x,y,z,radius)
    If add_periodic: particles that cross the domain boundary will be add.
    If trim: particles that do not cross the subdomain boundary will be deleted.
    """

    return PySphereList(spheres,radii)