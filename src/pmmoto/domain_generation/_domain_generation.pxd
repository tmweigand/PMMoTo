import cython

import numpy as np
from numpy cimport int8_t, int16_t, int32_t, int64_t
from numpy cimport uint8_t, uint16_t, uint32_t, uint64_t

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr


cdef extern from "sphere_pack.hpp":
	cdef struct Coords:
		double x
		double y
		double z

	cdef cppclass Sphere:
		Coords coordinates
		double radius

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

	unique_ptr[SphereList] initialize_sphere_list(
		vector[vector[double]] sphere_data,
		vector[double] point,
		double radius,
		bool kd_tree,
		bool trim
	)


cdef extern from "domain_generation.hpp":
	cdef struct Grid:
		vector[double] x
		vector[double] y
		vector[double] z
		vector[size_t] strides

	cdef struct Verlet:
		size_t num_verlet
		vector[vector[double]] centroids
		vector[double] diameters
		vector[vector[vector[size_t]]] loops


	cdef void gen_sphere_img_brute_force(
		uint8_t *img,
		Grid grid,
		Verlet verlet,
		unique_ptr[SphereList] spherelist
	)

	cdef void gen_sphere_img_kd_method(
		uint8_t *img,
		Grid grid,
		Verlet verlet,
		unique_ptr[SphereList] spherelist
	)