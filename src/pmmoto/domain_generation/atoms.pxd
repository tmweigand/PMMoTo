"""atoms.pxd"""

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

cdef extern from "atoms.hpp":
    
	cdef cppclass Atom

	cdef cppclass AtomList:
		vector[Atom] atoms
		AtomList(
			const vector[vector[double]]& atom_data, 
			bool build_kd
			)
		size_t size()
		vector[double] collect_kd_distances(
			vector[double] point,
			double radius
		)

	cdef shared_ptr[AtomList] initialize_list[AtomList,Atom](
		vector[vector[double]] data,
		vector[double] point,
		double radius,
		bool kd_tree,
		bool trim
	)