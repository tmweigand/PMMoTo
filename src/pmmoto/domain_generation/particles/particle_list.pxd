"""particles.pxd"""

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

cdef extern from "particle_list.hpp":

	cdef cppclass ParticleList

	cdef shared_ptr[ParticleList] initialize_list(
		vector[vector[double]] data,
		vector[double] point,
		double radius,
		vector[vector[double]] box,
		bool kd_tree,
		bool trim
	)

	cdef vector[vector[double]] return_particles(
		shared_ptr[ParticleList]
	)