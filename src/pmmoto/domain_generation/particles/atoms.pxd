"""atoms.pxd"""

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

cdef extern from "atoms.hpp":
    
	cdef cppclass Atom

	cdef cppclass AtomList:
		
		AtomList(
			const vector[vector[double]]& atom_data, 
			bool build_kd
			)

		vector[double] collect_kd_distances(
			vector[double] point,
			double radius
		)

	cdef shared_ptr[AtomList] initialize_list[AtomList,Atom](
		vector[vector[double]] data,
		vector[vector[double]] domain_box,
		vector[vector[double]] subdomain_box,
		bool add_periodic
	)

	cdef vector[vector[double]] return_particles[AtomList](
		shared_ptr[AtomList],
		bool return_own
	)

	cdef void set_own_particles[AtomList](
		shared_ptr[AtomList],
		vector[vector[double]] subdomain_box,
	)

