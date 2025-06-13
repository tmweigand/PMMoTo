"""atoms.pxd"""

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp.unordered_map cimport unordered_map

from .particle_list cimport Box

cdef extern from "atoms.hpp":

	cdef cppclass AtomList:

		int label
		double radius

		AtomList(
			vector[vector[double]] coordinates,
			double radii,
			int label,
			double mass
		) except +

		vector[vector[double]] return_atoms(
			bool return_own,
			bool return_label
		)

		vector[vector[double]] get_coordinates()

		size_t get_atom_count()

		size_t size()

		void build_KDtree()

		void add_periodic_atoms(
			Box domain_box,
			Box subdomain_box
		)

		void set_own_atoms(
			Box subdomain_box
		)

		void trim_atoms_intersecting(
			Box subdomain_box
		)

		void trim_atoms_within(
			Box subdomain_box
		)

		vector[double] collect_kd_distances(
			vector[double] point,
			double radius
		)

	cdef unordered_map[int, vector[vector[double]]] group_atoms_by_type(
		vector[vector[double]] atom_coordinates,
        vector[int] atom_ids
		)


	cdef vector[double] atom_id_to_values(
		vector[int] atom_ids,
    	unordered_map[int, double] values)