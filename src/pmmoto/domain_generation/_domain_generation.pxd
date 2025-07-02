import cython

import numpy as np
from numpy cimport int8_t, int16_t, int32_t, int64_t
from numpy cimport uint8_t, uint16_t, uint32_t, uint64_t

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.memory cimport shared_ptr

from ..particles.particle_list cimport Box
from ..particles.spheres cimport SphereList
from ..particles.cylinders cimport CylinderList

cdef extern from "domain_generation.hpp":
	cdef struct Grid:
		vector[double] x
		vector[double] y
		vector[double] z
		vector[size_t] strides

	cdef struct Verlet:
		size_t num_verlet
		vector[vector[vector[size_t]]] loops
		vector[vector[double]] centroids
		vector[double] diameters
		unordered_map[int,Box] box


	cdef void gen_sphere_img_brute_force(
		uint8_t *img,
		Grid grid,
		Verlet verlet,
		shared_ptr[SphereList] shape_list
	)

	cdef void gen_sphere_img_brute_force(
		uint8_t *img,
		Grid grid,
		Verlet verlet,
		shared_ptr[CylinderList] shape_list
	)


	cdef void gen_sphere_img_kd_method(
		uint8_t *img,
		Grid grid,
		Verlet verlet,
		shared_ptr[SphereList] spherelist
	)