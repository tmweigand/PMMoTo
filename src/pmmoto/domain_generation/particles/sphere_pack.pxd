"""sphere_pack.pxd"""
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

cdef extern from "sphere_pack.hpp":

	cdef cppclass Sphere

	cdef cppclass SphereList:
		vector[Sphere] spheres
		SphereList(
			const vector[vector[double]]& sphere_data, 
			bool build_kd
			)
		size_t size()

	cdef vector[Sphere] trim_sphere_list_spheres(
		vector[vector[double]] sphere_data,
		vector[double] point,
		double radius,
		bool kd_tree
	)

	cdef shared_ptr[SphereList] initialize_list[SphereList,Sphere](
		vector[vector[double]] data,
		vector[vector[double]] domain_box,
		vector[vector[double]] subdomain_box,
		bool add_periodic
	)

	cdef vector[vector[double]] return_particles[SphereList](
		shared_ptr[SphereList],
		bool return_own
	)

	cdef void set_own_particles[SphereList](
		shared_ptr[SphereList],
		vector[vector[double]] subdomain_box,
	)