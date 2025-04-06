"""spheres.pxd"""
from libcpp cimport bool
from libcpp.vector cimport vector

from .particle_list cimport Box

cdef extern from "spheres.hpp":

	cdef cppclass SphereList:
		
		SphereList(
            double radii
		) except +

		SphereList() except +
		
		SphereList(
			vector[vector[double]] coordinates,
			vector[double] radii,
		) except +

		void build_KDtree()

		size_t size()

		size_t get_own_count()

		void set_masses(vector[double] masses)

		vector[double] get_masses()

		vector[vector[double]] get_coordinates()

		vector[vector[double]] return_spheres(
			bool return_own,
		)

		void add_periodic_spheres(
			Box domain_box,
			Box subdomain_box
		)

		void own_spheres(
			Box subdomain_box
		)

		void trim_spheres_intersecting(
			Box subdomain_box
		)

		void trim_spheres_within(
			Box subdomain_box
		)